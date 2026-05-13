#!/usr/bin/env python3
"""Download public Cell Painting datasets and save illumination-corrected PNGs."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as scipy_io

from cellpainting_io import (
    download_url,
    extract_tar_members,
    list_zip_files,
    read_s3_bytes,
    read_zip_member,
    unsigned_s3_client,
)
from illumination import (
    correct_bbbc036_to_uint8,
    correct_cpg0016_to_uint8,
    image_bytes_to_array,
    save_uint8_png,
)
from manifests import (
    BBBC036_CHANNELS,
    BBBC036_GROUND_TRUTH_URL,
    BBBC036_ILLUM_URL_TEMPLATE,
    BBBC036_IMAGE_URL_TEMPLATE,
    BBBC036_PLATES,
    CPG_CHANNELS,
    CPG_LOAD_DATA_TEMPLATE,
    CPG_METADATA_REPO_URL,
    parse_plate_args,
    read_plates_file,
)

CPG_METADATA_FILES = (
    "plate.csv.gz",
    "well.csv.gz",
    "compound.csv.gz",
    "compound_source.csv.gz",
    "orf.csv.gz",
    "crispr.csv.gz",
    "microscope_config.csv",
    "microscope_filter.csv",
    "cellprofiler_version.csv",
)

BBBC036_METADATA_COLUMNS = (
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Assay_Plate_Barcode",
    "Metadata_Plate_Map_Name",
    "Metadata_well_position",
    "Metadata_ASSAY_WELL_ROLE",
    "Metadata_broad_sample",
    "Metadata_mmoles_per_liter",
    "Metadata_solvent",
    "Metadata_pert_id",
    "Metadata_pert_mfc_id",
    "Metadata_pert_well",
    "Metadata_pert_id_vendor",
    "Metadata_cell_id",
    "Metadata_broad_sample_type",
    "Metadata_pert_vehicle",
    "Metadata_pert_type",
)


def _metadata_cache(args: argparse.Namespace, dataset: str) -> Path:
    return args.metadata_root / dataset


def _ensure_cpg_metadata_repo(args: argparse.Namespace) -> Path:
    if args.cpg_metadata_repo:
        repo = args.cpg_metadata_repo
    else:
        repo = _metadata_cache(args, "cpg0016") / "datasets"
        if not repo.exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", CPG_METADATA_REPO_URL, str(repo)],
                check=True,
            )

    required = [repo / "metadata" / "plate.csv.gz", repo / "metadata" / "well.csv.gz"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CPG metadata files: {missing}")
    return repo


def _cache_cpg_benchmark_metadata(args: argparse.Namespace, repo: Path) -> None:
    """Copy public JUMP metadata files into this install cache."""
    output_dir = _metadata_cache(args, "cpg0016") / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in CPG_METADATA_FILES:
        source = repo / "metadata" / filename
        if not source.exists():
            continue
        target = output_dir / filename
        if args.overwrite or not target.exists():
            shutil.copy2(source, target)


def _select_cpg_plates(args: argparse.Namespace, plate_df: pd.DataFrame) -> list[str]:
    selected = parse_plate_args(args.plates)
    if args.plates_file:
        selected.extend(read_plates_file(args.plates_file))
    if not selected:
        selected = sorted(plate_df["Metadata_Plate"].astype(str).unique().tolist())
    selected = sorted(dict.fromkeys(str(plate) for plate in selected))
    if args.max_plates is not None:
        selected = selected[: args.max_plates]
    available = set(plate_df["Metadata_Plate"].astype(str))
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"Selected CPG plates missing from metadata: {missing[:10]}")
    return selected


def _load_cpg_plate_table(row: pd.Series, wells: pd.DataFrame, s3_client) -> pd.DataFrame:
    row_dict = row.to_dict()
    candidates = [
        CPG_LOAD_DATA_TEMPLATE.format(**row_dict, filename="load_data_with_illum.parquet"),
        CPG_LOAD_DATA_TEMPLATE.format(**row_dict, filename="load_data_with_illum.csv"),
        CPG_LOAD_DATA_TEMPLATE.format(**row_dict, filename="load_data.csv"),
    ]
    errors = []
    for s3_uri in candidates:
        try:
            payload = read_s3_bytes(s3_uri, client=s3_client)
            if s3_uri.endswith(".parquet"):
                plate_df = pd.read_parquet(BytesIO(payload))
            else:
                plate_df = pd.read_csv(BytesIO(payload))
            break
        except Exception as error:
            errors.append(f"{s3_uri}: {error}")
    else:
        raise FileNotFoundError("Could not load CPG load_data file:\n" + "\n".join(errors))

    for frame in (plate_df, wells):
        for column in ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]:
            frame[column] = frame[column].astype(str)
    plate_df = plate_df.merge(wells, on=["Metadata_Source", "Metadata_Plate", "Metadata_Well"])
    if "Metadata_Site" in plate_df.columns:
        plate_df["FOV"] = plate_df["Metadata_Site"].astype(int)
    else:
        plate_df["FOV"] = plate_df.groupby("Metadata_Well").cumcount().add(1)
    return plate_df


def _cache_cpg_plate_load_data(args: argparse.Namespace, plate: str, plate_df: pd.DataFrame) -> None:
    output_dir = _metadata_cache(args, "cpg0016") / "load_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{plate}_load_data_with_illum.csv"
    if args.overwrite or not output_path.exists():
        plate_df.to_csv(output_path, index=False)


def _cache_cpg_illumination(plate_df: pd.DataFrame, plate: str, illum_dir: Path, s3_client) -> dict[str, np.ndarray]:
    illum_dir.mkdir(parents=True, exist_ok=True)
    illuminations = {}
    first = plate_df.iloc[0]
    for channel in CPG_CHANNELS:
        local_path = illum_dir / f"{channel.name}_illum.npy"
        if not local_path.exists():
            local_path.write_bytes(read_s3_bytes(first[channel.illum_url_column], client=s3_client))
        illuminations[channel.name] = np.load(local_path)
    return illuminations


def _process_cpg_site(
    row: pd.Series,
    plate_output_dir: Path,
    illuminations: dict[str, np.ndarray],
    s3_client,
    overwrite: bool,
) -> dict[str, Any]:
    written = []
    for channel in CPG_CHANNELS:
        filename = f"{row.Metadata_Well}_i{int(row.FOV)}_ch{channel.output_index}.png"
        output_path = plate_output_dir / filename
        if output_path.exists() and not overwrite:
            written.append(filename)
            continue

        image = image_bytes_to_array(read_s3_bytes(row[channel.raw_url_column], client=s3_client))
        corrected = correct_cpg0016_to_uint8(image, illuminations[channel.name])
        tmp_path = output_path.with_suffix(".png.tmp")
        save_uint8_png(tmp_path, corrected)
        os.replace(tmp_path, output_path)
        written.append(filename)

    return {
        "Metadata_Source": row.Metadata_Source,
        "Metadata_Batch": row.Metadata_Batch,
        "Metadata_Plate": row.Metadata_Plate,
        "Metadata_Well": row.Metadata_Well,
        "FOV": int(row.FOV),
        "Metadata_JCP2022": row.Metadata_JCP2022,
        "files": ";".join(written),
        "status": "ok",
    }


def install_cpg0016(args: argparse.Namespace) -> None:
    if args.cpg_output_root is None:
        raise ValueError("--cpg-output-root is required for cpg0016")

    repo = _ensure_cpg_metadata_repo(args)
    _cache_cpg_benchmark_metadata(args, repo)
    plates_df = pd.read_csv(repo / "metadata" / "plate.csv.gz")
    wells = pd.read_csv(repo / "metadata" / "well.csv.gz")
    plates = _select_cpg_plates(args, plates_df)
    s3_client = unsigned_s3_client()

    for plate in plates:
        plate_row = plates_df[plates_df["Metadata_Plate"].astype(str) == plate].iloc[0]
        plate_df = _load_cpg_plate_table(plate_row, wells, s3_client)
        _cache_cpg_plate_load_data(args, plate, plate_df)
        if args.max_wells is not None:
            keep_wells = sorted(plate_df["Metadata_Well"].unique())[: args.max_wells]
            plate_df = plate_df[plate_df["Metadata_Well"].isin(keep_wells)].reset_index(drop=True)

        plate_output_dir = args.cpg_output_root / plate
        plate_output_dir.mkdir(parents=True, exist_ok=True)
        illuminations = _cache_cpg_illumination(
            plate_df,
            plate,
            _metadata_cache(args, "cpg0016") / "illum" / plate,
            s3_client,
        )

        print(f"CPG0016 {plate}: correcting {len(plate_df)} sites with {args.workers} workers")
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    _process_cpg_site,
                    row,
                    plate_output_dir,
                    illuminations,
                    s3_client,
                    args.overwrite,
                )
                for _, row in plate_df.iterrows()
            ]
            for future in as_completed(futures):
                rows.append(future.result())

        manifest_path = _metadata_cache(args, "cpg0016") / "manifests" / f"{plate}.csv"
        _write_manifest(manifest_path, rows)
        print(f"CPG0016 {plate}: wrote {len(rows)} manifest rows -> {manifest_path}")


def _select_bbbc036_plates(args: argparse.Namespace) -> list[str]:
    selected = parse_plate_args(args.plates)
    if args.plates_file:
        selected.extend(read_plates_file(args.plates_file))
    if not selected:
        selected = list(BBBC036_PLATES)
    selected = sorted(dict.fromkeys(str(plate) for plate in selected))
    if args.max_plates is not None:
        selected = selected[: args.max_plates]
    available = set(BBBC036_PLATES)
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"Selected BBBC036 plates not in built-in plate list: {missing[:10]}")
    return selected


def _find_bbbc036_illumination_files(extract_dir: Path, plate: str) -> dict[str, Path]:
    result = {}
    for channel in BBBC036_CHANNELS:
        matches = sorted(extract_dir.rglob(f"{plate}_Illum{channel.illum_name}.mat"))
        if not matches:
            raise FileNotFoundError(f"Missing BBBC036 illumination file for plate={plate}, channel={channel.name}")
        result[channel.name] = matches[0]
    return result


def _find_bbbc036_plate_metadata_file(root: Path, plate: str) -> Path | None:
    candidates = [
        root / f"Plate_{plate}" / "profiles" / "mean_well_profiles.csv",
        root / plate / "profiles" / "mean_well_profiles.csv",
        root / "profiles" / "mean_well_profiles.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob("profiles/mean_well_profiles.csv"))
    return matches[0] if matches else None


def _find_bbbc036_qc_file(root: Path, plate: str) -> Path | None:
    candidates = [
        root / f"Plate_{plate}" / "quality_control" / "qc.csv",
        root / plate / "quality_control" / "qc.csv",
        root / "quality_control" / "qc.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob("quality_control/qc.csv"))
    return matches[0] if matches else None


def _ensure_bbbc036_ground_truth(args: argparse.Namespace) -> Path:
    output = _metadata_cache(args, "bbbc036") / "metadata" / "BBBC036_v1_DatasetGroundTruth.csv"
    if args.bbbc036_ground_truth:
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not output.exists():
            shutil.copy2(args.bbbc036_ground_truth, output)
        return output
    if not output.exists() or args.overwrite:
        download_url(BBBC036_GROUND_TRUTH_URL, output, overwrite=True)
    return output


def _cache_bbbc036_plate_metadata(args: argparse.Namespace, plate: str, source_root: Path) -> None:
    """Cache per-plate BBBC036 benchmark metadata with MoA annotations."""
    metadata_file = _find_bbbc036_plate_metadata_file(source_root, plate)
    if metadata_file is None:
        print(f"BBBC036 {plate}: no mean_well_profiles.csv found for metadata cache")
        return

    output_dir = _metadata_cache(args, "bbbc036") / "metadata" / "per_plate"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{plate}_metadata.parquet"
    if output_path.exists() and not args.overwrite:
        return

    available_columns = pd.read_csv(metadata_file, nrows=0).columns.tolist()
    use_columns = [column for column in BBBC036_METADATA_COLUMNS if column in available_columns]
    metadata = pd.read_csv(metadata_file, usecols=use_columns)
    metadata["Metadata_Plate"] = metadata["Metadata_Plate"].astype(str)

    ground_truth = pd.read_csv(_ensure_bbbc036_ground_truth(args))
    metadata = metadata.merge(
        ground_truth[["Metadata_broad_sample", "Metadata_moa", "Metadata_target"]],
        on="Metadata_broad_sample",
        how="left",
    )
    metadata.to_parquet(output_path, index=False)

    qc_file = _find_bbbc036_qc_file(source_root, plate)
    if qc_file is not None:
        qc_output = output_dir / f"{plate}_qc.csv"
        if args.overwrite or not qc_output.exists():
            shutil.copy2(qc_file, qc_output)


def _ensure_bbbc036_illumination(args: argparse.Namespace, plate: str) -> dict[str, np.ndarray]:
    cache = _metadata_cache(args, "bbbc036")
    illum_dir = cache / "illum" / plate
    illum_arrays = {}
    existing = True
    for channel in BBBC036_CHANNELS:
        npy_path = illum_dir / f"{channel.name}_illum.npy"
        if not npy_path.exists():
            existing = False
            break
    if not existing:
        if args.bbbc036_illum_root:
            files = _find_bbbc036_illumination_files(args.bbbc036_illum_root, plate)
            _cache_bbbc036_plate_metadata(args, plate, args.bbbc036_illum_root)
            archive = None
            extract_dir = None
        else:
            archive = cache / "archives" / f"Plate_{plate}.tar.gz"
            url = BBBC036_ILLUM_URL_TEMPLATE.format(plate=plate)
            print(f"BBBC036 {plate}: downloading illumination archive")
            download_url(url, archive, overwrite=args.overwrite)
            extract_dir = cache / "extracted" / plate
            if extract_dir.exists() and args.overwrite:
                shutil.rmtree(extract_dir)
            extract_tar_members(archive, extract_dir, member_token="illumination_correction_functions")
            extract_tar_members(archive, extract_dir, member_token="profiles/mean_well_profiles.csv")
            extract_tar_members(archive, extract_dir, member_token="quality_control/qc.csv")
            files = _find_bbbc036_illumination_files(extract_dir, plate)
            _cache_bbbc036_plate_metadata(args, plate, extract_dir)

        illum_dir.mkdir(parents=True, exist_ok=True)
        for channel in BBBC036_CHANNELS:
            image = scipy_io.loadmat(files[channel.name])["Image"]
            np.save(illum_dir / f"{channel.name}_illum.npy", image)
        if archive is not None and not args.keep_archives and archive.exists():
            archive.unlink()
        if extract_dir is not None and not args.keep_archives and extract_dir.exists():
            shutil.rmtree(extract_dir)

    for channel in BBBC036_CHANNELS:
        illum_arrays[channel.name] = np.load(illum_dir / f"{channel.name}_illum.npy")
    return illum_arrays


def _process_bbbc036_channel(
    plate: str,
    channel_name: str,
    illumination: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    assert args.bbbc036_output_root is not None
    plate_output_dir = args.bbbc036_output_root / plate
    plate_output_dir.mkdir(parents=True, exist_ok=True)

    archive = None
    local_dir = None
    if args.bbbc036_raw_unzipped_root:
        local_dir = args.bbbc036_raw_unzipped_root / f"{plate}-{channel_name}"
        if not local_dir.exists():
            raise FileNotFoundError(f"Missing local BBBC036 raw channel directory: {local_dir}")
        members = sorted(str(path) for path in local_dir.glob("*.tif"))
    else:
        cache = _metadata_cache(args, "bbbc036")
        archive = cache / "raw_zips" / f"{plate}-{channel_name}.zip"
        url = BBBC036_IMAGE_URL_TEMPLATE.format(plate=plate, channel=channel_name)
        download_url(url, archive, overwrite=args.overwrite)
        members = list_zip_files(archive, suffix=".tif")

    if args.max_images_per_channel is not None:
        members = members[: args.max_images_per_channel]

    rows = []
    for member in members:
        output_name = f"{Path(member).stem}_ch_{channel_name}.png"
        output_path = plate_output_dir / output_name
        if output_path.exists() and not args.overwrite:
            rows.append({"plate": plate, "channel": channel_name, "source": member, "filename": output_name, "status": "ok"})
            continue

        if local_dir is not None:
            image_payload = Path(member).read_bytes()
            source_name = Path(member).name
        else:
            assert archive is not None
            image_payload = read_zip_member(archive, member)
            source_name = member
        image = image_bytes_to_array(image_payload)
        corrected = correct_bbbc036_to_uint8(image, illumination)
        tmp_path = output_path.with_suffix(".png.tmp")
        save_uint8_png(tmp_path, corrected)
        os.replace(tmp_path, output_path)
        rows.append({"plate": plate, "channel": channel_name, "source": source_name, "filename": output_name, "status": "ok"})

    if archive is not None and not args.keep_archives and archive.exists():
        archive.unlink()
    return rows


def install_bbbc036(args: argparse.Namespace) -> None:
    if args.bbbc036_output_root is None:
        raise ValueError("--bbbc036-output-root is required for bbbc036")

    plates = _select_bbbc036_plates(args)
    for plate in plates:
        illuminations = _ensure_bbbc036_illumination(args, plate)
        print(f"BBBC036 {plate}: correcting channels with {args.workers} workers")
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(_process_bbbc036_channel, plate, channel.name, illuminations[channel.name], args)
                for channel in BBBC036_CHANNELS
            ]
            for future in as_completed(futures):
                rows.extend(future.result())
        manifest_path = _metadata_cache(args, "bbbc036") / "manifests" / f"{plate}.csv"
        _write_manifest(manifest_path, rows)
        print(f"BBBC036 {plate}: wrote {len(rows)} manifest rows -> {manifest_path}")


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_outputs(args: argparse.Namespace) -> None:
    from PIL import Image

    roots = []
    if args.dataset in {"cpg0016", "all"} and args.cpg_output_root is not None:
        roots.append(args.cpg_output_root)
    if args.dataset in {"bbbc036", "all"} and args.bbbc036_output_root is not None:
        roots.append(args.bbbc036_output_root)
    for root in roots:
        images = sorted(root.glob("*/*.png"))
        if not images:
            raise ValueError(f"No PNG outputs found under {root}")
        with Image.open(images[0]) as image:
            array = np.asarray(image)
            if array.dtype != np.uint8:
                raise ValueError(f"Expected uint8 image, got {array.dtype}: {images[0]}")
            print(f"Validated {images[0]}: mode={image.mode}, size={image.size}, dtype={array.dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cpg0016", "bbbc036", "all"], required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--cpg-output-root", type=Path)
    parser.add_argument("--bbbc036-output-root", type=Path)
    parser.add_argument("--cpg-metadata-repo", type=Path, help="Existing jump-cellpainting/datasets checkout.")
    parser.add_argument("--bbbc036-illum-root", type=Path, help="Existing extracted BBBC036 GigaDB/gigascience_upload root.")
    parser.add_argument("--bbbc036-raw-unzipped-root", type=Path, help="Existing BBBC036 raw TIFF root with <plate>-<channel>/ folders.")
    parser.add_argument("--bbbc036-ground-truth", type=Path, help="Existing BBBC036_v1_DatasetGroundTruth.csv; downloaded if omitted.")
    parser.add_argument("--plates", action="append", default=[], help="Plate(s) to process. Repeat or comma-separate.")
    parser.add_argument("--plates-file", type=Path)
    parser.add_argument("--max-plates", type=int)
    parser.add_argument("--max-wells", type=int, help="CPG smoke-test limit on wells per plate.")
    parser.add_argument("--max-images-per-channel", type=int, help="BBBC036 smoke-test limit per channel ZIP.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.metadata_root.mkdir(parents=True, exist_ok=True)
    if args.dataset in {"cpg0016", "all"}:
        install_cpg0016(args)
    if args.dataset in {"bbbc036", "all"}:
        install_bbbc036(args)
    if args.validate:
        validate_outputs(args)


if __name__ == "__main__":
    main()
