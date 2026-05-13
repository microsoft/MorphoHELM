#!/usr/bin/env python3
"""Convert raw Cell Painting plate folders into per-plate HuggingFace datasets.

This ports the JUMP_toy_example conversion pattern into the benchmark repo:
one raw image plate folder becomes one saved HuggingFace DatasetDict with a
`train` split. Each row keeps the image plus filename, plate, well, FOV, channel
metadata, and bounding boxes looked up in the same plate-level way used by the
toy scripts.
"""

from __future__ import annotations

import argparse
import ast
import io
import os
import pickle
import re
import shutil
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, Image, load_from_disk
from PIL import Image as PILImage


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CPG_RAW_ROOT = PROJECT_ROOT / "data" / "corrected_images" / "cpg0016"
DEFAULT_CPG_BBOX_ROOT = PROJECT_ROOT / "data" / "bounding_boxes" / "bounding_boxes_cpg"
DEFAULT_BBBC036_RAW_ROOT = PROJECT_ROOT / "data" / "corrected_images" / "bbbc036"
DEFAULT_BBBC036_BBOX_ROOT = PROJECT_ROOT / "data" / "bounding_boxes" / "bounding_boxes_bbbc036"

CPG_RE = re.compile(r"^(?P<well>[A-Za-z]{1,2}\d{2})_(?P<fov>i\d+)_ch(?P<channel>\d+)\.png$")
BBBC036_RE = re.compile(
    r"^.+?_(?P<well>[A-Za-z]{1,2}\d{2})_(?P<fov>s\d+)_w(?P<channel>\d)[^_]*_ch_(?P<stain>.+)\.png$"
)


@dataclass(frozen=True)
class ImageRecord:
    image: dict[str, Any]
    filename: str
    plate_name: str
    wells: str
    well: str
    fov: str
    channel: str
    bounding_boxes: list[Any]


def _is_plate_dir(path: Path) -> bool:
    name = path.name.lower()
    return path.is_dir() and not any(token in name for token in ("bf", "metadata", "brightfield"))


def discover_plates(raw_root: Path) -> list[str]:
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {raw_root}")
    return sorted(path.name for path in raw_root.iterdir() if _is_plate_dir(path))


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _normalise_boxes(boxes: Any) -> list[Any]:
    if boxes is None:
        return []
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    return list(boxes)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        if info.max <= 255:
            return np.clip(array, 0, 255).astype(np.uint8)
        scaled = array.astype(np.float32) / float(info.max)
        return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)
    finite = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if finite.size and finite.max() <= 1.0 and finite.min() >= 0.0:
        finite = finite * 255.0
    else:
        min_value = float(finite.min()) if finite.size else 0.0
        max_value = float(finite.max()) if finite.size else 0.0
        if max_value > min_value:
            finite = (finite - min_value) / (max_value - min_value) * 255.0
    return np.clip(np.rint(finite), 0, 255).astype(np.uint8)


def _uint8_png_payload(image_path: Path) -> dict[str, Any]:
    with PILImage.open(image_path) as image:
        image_uint8 = PILImage.fromarray(_to_uint8(np.asarray(image)))
        buffer = io.BytesIO()
        image_uint8.save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": image_path.name}


def _load_cpg_boxes(bbox_root: Path, plate: str) -> dict[str, list[Any]]:
    path = bbox_root / f"{plate}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing CPG bounding-box pickle: {path}")
    payload = load_pickle(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected CPG bounding-box schema in {path}")
    plate_payload = payload.get(plate, payload)
    if not isinstance(plate_payload, dict):
        raise ValueError(f"Unexpected CPG plate payload in {path}")
    return {str(key): _normalise_boxes(value) for key, value in plate_payload.items()}


def _bbbc036_well_fov_key(key: str) -> str:
    parts = key.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse BBBC036 bounding-box key: {key}")
    return f"{parts[1]}_{parts[2]}".lower()


def _load_bbbc036_boxes(bbox_root: Path, plate: str) -> dict[str, list[Any]]:
    path = bbox_root / f"{plate}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing BBBC036 bounding-box pickle: {path}")
    payload = load_pickle(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected BBBC036 bounding-box schema in {path}")

    boxes_by_well_fov: dict[str, list[Any]] = {}
    for key in sorted(payload):
        boxes = payload[key]
        # The toy script collapses long BBBC036 image IDs to "<well>_<site>".
        # Keep the first non-empty set deterministically after sorting keys.
        collapsed = _bbbc036_well_fov_key(str(key))
        if collapsed not in boxes_by_well_fov or (
            not boxes_by_well_fov[collapsed] and boxes
        ):
            boxes_by_well_fov[collapsed] = _normalise_boxes(boxes)
    return boxes_by_well_fov


def _parse_cpg_image(filename: str) -> tuple[str, str, str]:
    match = CPG_RE.match(filename)
    if not match:
        raise ValueError(f"Cannot parse CPG0016 image filename: {filename}")
    return match.group("well"), match.group("fov"), match.group("channel")


def _parse_bbbc036_image(filename: str) -> tuple[str, str, str]:
    match = BBBC036_RE.match(filename)
    if not match:
        raise ValueError(f"Cannot parse BBBC036 image filename: {filename}")
    return match.group("well").lower(), match.group("fov").lower(), match.group("channel")


def build_cpg_records(raw_root: Path, bbox_root: Path, plate: str, max_images: int | None) -> list[ImageRecord]:
    plate_dir = raw_root / plate
    boxes = _load_cpg_boxes(bbox_root, plate)
    image_paths = sorted(path for path in plate_dir.iterdir() if path.suffix.lower() == ".png")
    if max_images is not None:
        image_paths = image_paths[:max_images]
    records = []
    for image_path in image_paths:
        well, fov, channel = _parse_cpg_image(image_path.name)
        # Matches the toy script: all channels for a well/FOV use the ch1 boxes.
        bbox_key = f"{well}_{fov}_ch1.png"
        records.append(
            ImageRecord(
                image=_uint8_png_payload(image_path),
                filename=image_path.name,
                plate_name=plate,
                wells=well,
                well=well,
                fov=fov,
                channel=channel,
                bounding_boxes=boxes.get(bbox_key, []),
            )
        )
    return records


def build_bbbc036_records(raw_root: Path, bbox_root: Path, plate: str, max_images: int | None) -> list[ImageRecord]:
    plate_dir = raw_root / plate
    boxes = _load_bbbc036_boxes(bbox_root, plate)
    image_paths = sorted(path for path in plate_dir.iterdir() if path.suffix.lower() == ".png")
    if max_images is not None:
        image_paths = image_paths[:max_images]
    records = []
    for image_path in image_paths:
        well, fov, channel = _parse_bbbc036_image(image_path.name)
        bbox_key = f"{well}_{fov}"
        records.append(
            ImageRecord(
                image=_uint8_png_payload(image_path),
                filename=image_path.name,
                plate_name=plate,
                wells=well,
                well=well,
                fov=fov,
                channel=channel,
                bounding_boxes=boxes.get(bbox_key, []),
            )
        )
    return records


def records_to_dataset(records: list[ImageRecord]) -> DatasetDict:
    if not records:
        raise ValueError("No image records to save")
    data = {
        "image": [record.image for record in records],
        "filename": [record.filename for record in records],
        "plate_name": [record.plate_name for record in records],
        "wells": [record.wells for record in records],
        "well": [record.well for record in records],
        "fov": [record.fov for record in records],
        "channel": [record.channel for record in records],
        "bounding_boxes": [record.bounding_boxes for record in records],
    }
    dataset = Dataset.from_dict(data).cast_column("image", Image())
    return DatasetDict({"train": dataset})


def convert_plate(
    dataset_name: str,
    raw_root: Path,
    bbox_root: Path,
    output_root: Path,
    plate: str,
    overwrite: bool,
    max_images: int | None,
) -> dict[str, Any]:
    out_dir = output_root / plate
    if out_dir.exists():
        if not overwrite:
            return {"plate": plate, "status": "skipped", "output": str(out_dir)}
        shutil.rmtree(out_dir)

    if dataset_name == "cpg0016":
        records = build_cpg_records(raw_root, bbox_root, plate, max_images)
    elif dataset_name == "bbbc036":
        records = build_bbbc036_records(raw_root, bbox_root, plate, max_images)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    ds = records_to_dataset(records)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    with_boxes = sum(1 for record in records if record.bounding_boxes)
    return {
        "plate": plate,
        "status": "converted",
        "n_images": len(records),
        "n_with_boxes": with_boxes,
        "output": str(out_dir),
    }


def _worker(kwargs: dict[str, Any]) -> dict[str, Any]:
    return convert_plate(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw plate image folders to per-plate HuggingFace datasets.")
    parser.add_argument("--dataset", choices=["cpg0016", "bbbc036"], required=True)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--bbox-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plate", action="append", default=[], help="Plate name to convert. Can be repeated.")
    parser.add_argument("--plate-index", type=int, action="append", default=[], help="Convert plate by sorted index. Can be repeated.")
    parser.add_argument("--plates-file", type=Path, help="File containing a Python/list literal or one plate per line.")
    parser.add_argument("--max-plates", type=int, help="Convert only the first N selected/discovered plates.")
    parser.add_argument("--max-images", type=int, help="Smoke-test limit on images per plate.")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Load saved datasets and access the first image after conversion.")
    return parser.parse_args()


def _plates_from_file(path: Path) -> list[str]:
    text = path.read_text().strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return [line.strip() for line in text.splitlines() if line.strip()]


def select_plates(args: argparse.Namespace, raw_root: Path) -> list[str]:
    discovered = discover_plates(raw_root)
    selected: list[str] = []
    selected.extend(args.plate)
    selected.extend(discovered[index] for index in args.plate_index)
    if args.plates_file:
        selected.extend(_plates_from_file(args.plates_file))
    if not selected:
        selected = discovered
    selected = [str(plate) for plate in selected]
    missing = sorted(set(selected) - set(discovered))
    if missing:
        raise FileNotFoundError(f"Selected plates not found under {raw_root}: {missing[:10]}")
    selected = sorted(dict.fromkeys(selected))
    if args.max_plates is not None:
        selected = selected[: args.max_plates]
    if not selected:
        raise ValueError("No plates selected")
    return selected


def default_raw_root(dataset_name: str) -> Path:
    return DEFAULT_CPG_RAW_ROOT if dataset_name == "cpg0016" else DEFAULT_BBBC036_RAW_ROOT


def default_bbox_root(dataset_name: str) -> Path:
    return DEFAULT_CPG_BBOX_ROOT if dataset_name == "cpg0016" else DEFAULT_BBBC036_BBOX_ROOT


def validate_outputs(output_root: Path, plates: list[str]) -> None:
    for plate in plates:
        ds = load_from_disk(str(output_root / plate))
        train = ds["train"]
        if len(train) == 0:
            raise ValueError(f"Saved dataset is empty: {output_root / plate}")
        first = train[0]
        required = {"image", "filename", "plate_name", "wells", "fov", "bounding_boxes"}
        missing = required - set(first)
        if missing:
            raise ValueError(f"{plate}: missing fields {sorted(missing)}")
        if first["image"] is None:
            raise ValueError(f"{plate}: first image did not decode")
        if first["image"].mode not in {"L", "RGB", "RGBA"}:
            raise ValueError(f"{plate}: expected uint8-compatible image mode, got {first['image'].mode}")


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root or default_raw_root(args.dataset)
    bbox_root = args.bbox_root or default_bbox_root(args.dataset)
    plates = select_plates(args, raw_root)

    jobs = [
        {
            "dataset_name": args.dataset,
            "raw_root": raw_root,
            "bbox_root": bbox_root,
            "output_root": args.output_root,
            "plate": plate,
            "overwrite": args.overwrite,
            "max_images": args.max_images,
        }
        for plate in plates
    ]

    if args.num_workers > 1 and len(jobs) > 1:
        with Pool(processes=args.num_workers) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(job) for job in jobs]

    for result in results:
        if result["status"] == "converted":
            print(
                f"{result['plate']}: converted {result['n_images']} images "
                f"({result['n_with_boxes']} with boxes) -> {result['output']}"
            )
        else:
            print(f"{result['plate']}: skipped existing -> {result['output']}")

    if args.validate:
        validate_outputs(args.output_root, plates)
        print(f"Validated {len(plates)} saved dataset(s).")


if __name__ == "__main__":
    main()
