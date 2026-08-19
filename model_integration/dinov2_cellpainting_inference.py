#!/usr/bin/env python3
"""Run DINOv2_CellPainting inference over MorphoHELM corrected PNGs.

Faithful port of the DINOv2_CellPainting `inference.py` pipeline, adapted to read
MorphoHELM's per-channel corrected PNGs (instead of pre-merged uint16 TIFFs).

Pipeline (mirrors DINOv2_CellPainting/source + inference.py):
  1. Build the 5-channel FOV in the model TRAINING order [Mito, AGP, RNA, ER, DNA].
  2. `scale_intensities(img, 99.9)` -> per-channel percentile scaling to [0,1]
     (lower bound = 0.1 pct, upper = 99.9 pct; same as source/image_ops.py).
  3. Otsu threshold on the DNA channel (index 4); tile the FOV into `crop_size`
     crops (stride `stride`); drop crops whose DNA foreground fraction is below
     `min_area_ratio` (empty-crop exclusion). Same logic as
     source.image_ops.generate_cellcrops, made rectangle-safe for non-square FOVs
     (BBBC036 is 696x520); byte-identical to the original for square FOVs.
  4. `transforms.Normalize(CP_MEANS, CP_STDS)` on the stacked crops.
  5. Backbone CLS token per crop (source.inference_utils.forward_inference, which
     sets empty-crop embeddings to NaN), then pool ALL crops across ALL FOVs of a
     well via nanmean (source.inference_utils.aggregate_embeddings_plate).

Output: one aggregated well-level parquet per split, in MorphoHELM's schema:
  <outdir>/<split>/<model_name>_aggregated.parquet
  columns: Metadata_Plate, Metadata_Well, feat_0 ... feat_{D-1}
which is a drop-in for data-preprocessing/src/normalize_splits.py (it joins the
control metadata itself on Metadata_Plate + Metadata_Well).

Channel maps (MorphoHELM corrected PNGs -> model order [Mito,AGP,RNA,ER,DNA]):
  CPG0016  <well>_i<fov>_ch{0..4}.png  where ch0=AGP,1=DNA,2=ER,3=Mito,4=RNA
           -> stack [ch3, ch0, ch4, ch2, ch1]
  BBBC036  ..._<well>_s<fov>_w{1..5}..._ch_<stain>.png
           stain->slot: Mito=0, Ph_golgi=1, ERSytoBleed=2, ERSyto=3, Hoechst=4
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

MODEL_NAME_DEFAULT = "dino_v2_cellpainting"

# --- filename parsers (match raw_to_huggingface/transform_dataset.py) ---
CPG_RE = re.compile(r"^(?P<well>[A-Za-z]{1,2}\d{2})_i(?P<fov>\d+)_ch(?P<ch>\d)\.png$")
BBBC036_RE = re.compile(
    r"^.+?_(?P<well>[A-Za-z]{1,2}\d{2})_s(?P<fov>\d+)_w(?P<w>\d)[^_]*_ch_(?P<stain>.+)\.png$"
)

# model training order [Mito, AGP, RNA, ER, DNA]
CPG_MODEL_IDX = [3, 0, 4, 2, 1]  # index into ch0..ch4
BBBC_STAIN_TO_SLOT = {"Mito": 0, "Ph_golgi": 1, "ERSytoBleed": 2, "ERSyto": 3, "Hoechst": 4}


def _import_dino_source(dinov2_repo: str):
    """Import the DINOv2_CellPainting `source` helpers and `inference` module."""
    repo = os.path.abspath(os.path.expanduser(dinov2_repo))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import inference as dino_inf  # noqa: E402  (triggers repo runtime-env setup)
    from source import image_ops as imo  # noqa: E402
    from source.inference_utils import aggregate_embeddings_plate, forward_inference  # noqa: E402
    return dino_inf, imo, aggregate_embeddings_plate, forward_inference


def _generate_cellcrops_rect(img, crop_size, stride, imo, *, otsu_chan=4,
                             otsu_thresh_ratio=0.7, otsu_down_factor=10,
                             min_area_ratio=0.01, otsuth=None):
    """Rectangle-safe reimplementation of source.image_ops.generate_cellcrops.

    Identical output to the original for square FOVs; also supports non-square
    FOVs (the original assumes width == height). `img` is a (C, H, W) tensor in
    [0, 1]. Returns (list_of_crops, list_of_bool_labels).
    """
    img = torch.as_tensor(img)
    if otsuth is None:
        thresh = imo.pt_threshold_otsu(
            img[otsu_chan, 0:img.shape[1]:otsu_down_factor, 0:img.shape[2]:otsu_down_factor]
        )
    else:
        thresh = otsuth
    bin_img = img[otsu_chan, :, :] > otsu_thresh_ratio * thresh
    bin_patches = bin_img.unfold(0, crop_size, stride).unfold(1, crop_size, stride)
    img_patches = img.unfold(1, crop_size, stride).unfold(2, crop_size, stride)

    n_w, n_h = img_patches.shape[1], img_patches.shape[2]
    croplist, labels = [], []
    for r in range(n_w):
        for c in range(n_h):
            crop = img_patches[:, r, c, ...]
            bin_crop = bin_patches[r, c, ...]
            ratio = bin_crop.sum() / (bin_crop.shape[0] * bin_crop.shape[1])
            croplist.append(crop)
            labels.append(bool(ratio >= min_area_ratio))
    if len(labels) and np.sum(labels) == 0:
        labels[len(labels) // 2] = True
    return croplist, labels


def _discover_fovs(plate_dir: str, dataset: str) -> List[Tuple[str, str, Dict[int, str]]]:
    """Return [(well, fov, {model_slot: png_path})] for a plate directory.

    Well IDs keep their native filename case: CPG0016 is uppercase ("A01"),
    BBBC036 is lowercase ("a01") — matching each dataset's benchmark metadata.
    """
    groups: Dict[Tuple[str, str], Dict[int, str]] = {}
    for name in os.listdir(plate_dir):
        if not name.endswith(".png"):
            continue
        if dataset == "cpg0016":
            m = CPG_RE.match(name)
            if not m:
                continue
            # model slot s uses channel CPG_MODEL_IDX[s]; invert to map ch -> slot
            slot = CPG_MODEL_IDX.index(int(m.group("ch")))
            key = (m.group("well"), m.group("fov"))
        else:  # bbbc036
            m = BBBC036_RE.match(name)
            if not m:
                continue
            stain = m.group("stain")
            if stain not in BBBC_STAIN_TO_SLOT:
                continue
            slot = BBBC_STAIN_TO_SLOT[stain]
            key = (m.group("well"), m.group("fov"))
        groups.setdefault(key, {})[slot] = os.path.join(plate_dir, name)

    fovs = []
    for (well, fov), slots in sorted(groups.items()):
        if len(slots) == 5:
            fovs.append((well, fov, slots))
    return fovs


class PlateFovDataset(Dataset):
    """One item per FOV -> dict(crops=(Ncrops,C,cs,cs), labels, Metadata_Plate, Metadata_Well)."""

    def __init__(self, plate: str, fovs, imo, normalize, *, crop_size, stride,
                 scale_pct=99.9, min_area_ratio=0.01, otsu_chan=4):
        self.plate = plate
        self.fovs = fovs
        self.imo = imo
        self.normalize = normalize
        self.crop_size = crop_size
        self.stride = stride
        self.scale_pct = scale_pct
        self.min_area_ratio = min_area_ratio
        self.otsu_chan = otsu_chan

    def __len__(self):
        return len(self.fovs)

    def __getitem__(self, idx):
        well, _fov, slots = self.fovs[idx]
        chans = [np.asarray(Image.open(slots[s])).astype(np.float32) for s in range(5)]
        img = np.stack(chans, axis=0)  # (5, H, W) in model order
        img = self.imo.scale_intensities(img, self.scale_pct)  # per-channel -> [0,1]
        croplist, labels = _generate_cellcrops_rect(
            img, self.crop_size, self.stride, self.imo,
            otsu_chan=self.otsu_chan, min_area_ratio=self.min_area_ratio, otsuth=None,
        )
        crops = torch.stack(croplist)  # (Ncrops, C, cs, cs)
        crops = self.normalize(crops)
        return {
            "crops": crops,
            "labels": torch.BoolTensor(labels),
            "Metadata_Plate": self.plate,
            "Metadata_Well": well,
        }


def _build_backbone(args, dino_inf):
    cfg_ns = SimpleNamespace(
        config_file=args.config_file or "",
        opts=list(args.opts or []),
        arch=args.arch,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        img_size=args.img_size,
        ffn_layer=args.ffn_layer,
        block_chunks=args.block_chunks,
        num_register_tokens=args.num_register_tokens,
    )
    backbone = dino_inf.build_backbone_for_inference(cfg_ns)
    dino_inf.load_teacher_backbone(args.ckpt, backbone, strict=args.strict_backbone)
    embed_dim = getattr(backbone, "embed_dim", 384)
    plate_projection = dino_inf.extract_plate_projection_from_checkpoint(args.ckpt, embed_dim)
    return backbone, plate_projection


def _resolve_plates(args) -> List[str]:
    root = args.corrected_root
    if args.plates_file:
        with open(args.plates_file) as fh:
            plates = [ln.strip() for ln in fh if ln.strip()]
    elif args.plates:
        plates = [p for chunk in args.plates for p in chunk.split(",") if p.strip()]
    else:
        plates = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    plates = [p for p in plates if os.path.isdir(os.path.join(root, p))]
    exclude = set()
    for chunk in (args.exclude_plates or []):
        exclude.update(x for x in chunk.split(",") if x.strip())
    if exclude:
        plates = [p for p in plates if p not in exclude]
    if args.max_plates:
        plates = plates[: args.max_plates]
    return plates


def run(args) -> None:
    dino_inf, imo, aggregate_embeddings_plate, forward_inference = _import_dino_source(args.dinov2_repo)

    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    backbone, plate_projection = _build_backbone(args, dino_inf)
    embednet = dino_inf.DinoV2EmbeddingNet(backbone, plate_projection=plate_projection)
    if torch.cuda.is_available() and len(args.gpus) > 1:
        embednet = torch.nn.DataParallel(embednet, device_ids=args.gpus)
    embednet.to(device).eval()

    normalize = transforms.Normalize(torch.tensor(dino_inf.CP_MEANS), torch.tensor(dino_inf.CP_STDS))

    out_dir = os.path.join(args.outdir, args.split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.model_name}_aggregated.parquet")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] exists: {out_path}")
        return

    plates = _resolve_plates(args)
    print(f"{args.split}: {len(plates)} plate(s) from {args.corrected_root}")

    partial_dir = os.path.join(out_dir, "_partial", args.model_name)
    os.makedirs(partial_dir, exist_ok=True)

    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        emb_cols = [c for c in df.columns if c.startswith("emb")]
        rename = {c: f"feat_{i}" for i, c in enumerate(emb_cols)}
        df = df.rename(columns=rename)
        df["Metadata_Plate"] = df["Metadata_Plate"].astype(str)
        df["Metadata_Well"] = df["Metadata_Well"].astype(str)
        return df[["Metadata_Plate", "Metadata_Well"] + [rename[c] for c in emb_cols]]

    for plate in tqdm(plates, desc=f"{args.dataset}:{args.split} plates", unit="plate"):
        ppath = os.path.join(partial_dir, f"{plate}.parquet")
        if os.path.exists(ppath) and not args.overwrite:
            continue  # resumable: this plate already done
        plate_dir = os.path.join(args.corrected_root, plate)
        fovs = _discover_fovs(plate_dir, args.dataset)
        if args.max_fovs:
            fovs = fovs[: args.max_fovs]
        if not fovs:
            print(f"[warn] no complete 5-channel FOVs in {plate_dir}; skipping")
            continue

        ds = PlateFovDataset(plate, fovs, imo, normalize,
                             crop_size=args.size, stride=args.stride or args.size)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                            shuffle=False, drop_last=False)
        fov_df = pd.DataFrame(
            [{"Metadata_Plate": p, "Metadata_Well": w} for (w, _f, _s), p in
             zip(fovs, [plate] * len(fovs))]
        )

        plate_embs = []
        for batch in tqdm(loader, desc=f"crops {plate}", leave=False):
            with torch.no_grad():
                emb = forward_inference(embednet, batch["crops"], batch["labels"], device)
            plate_embs.append(emb.cpu().numpy())

        well_df = aggregate_embeddings_plate(
            plate_dfr=fov_df, plate_embs=plate_embs,
            my_cols=["Metadata_Plate", "Metadata_Well"], operation=args.operation,
        )
        # checkpoint this plate immediately (atomic write via temp + rename)
        tmp = ppath + ".tmp"
        _finalize(well_df).to_parquet(tmp, index=False)
        os.replace(tmp, ppath)

    # combine per-plate checkpoints into the final aggregated parquet
    part_paths = [os.path.join(partial_dir, f"{p}.parquet") for p in plates]
    part_paths = [p for p in part_paths if os.path.exists(p)]
    if not part_paths:
        raise RuntimeError("No embeddings produced (no plates/FOVs found).")
    result = pd.concat([pd.read_parquet(p) for p in part_paths], ignore_index=True)
    result = result.drop_duplicates(subset=["Metadata_Plate", "Metadata_Well"], keep="first")
    result.to_parquet(out_path, index=False)
    n_feat = sum(c.startswith("feat_") for c in result.columns)
    print(f"[ok] wrote {out_path}: {result.shape[0]} wells x {n_feat} features "
          f"({len(part_paths)} plates)")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["cpg0016", "bbbc036"])
    p.add_argument("--split", required=True,
                   help="Output split name, e.g. cpg-moa, cpg-crispr, bbbc036.")
    p.add_argument("--corrected-root", required=True,
                   help="Plate-folder root, e.g. $ROOT/corrected/cpg0016.")
    p.add_argument("--outdir", required=True, help="Aggregated-parquet output root.")
    p.add_argument("--model-name", default=MODEL_NAME_DEFAULT)
    p.add_argument("--plates", nargs="*", default=None, help="Plate IDs (repeat or comma-separated).")
    p.add_argument("--plates-file", default=None, help="File with one plate ID per line.")
    p.add_argument("--exclude-plates", nargs="*", default=None,
                   help="Plate IDs to skip (benchmark excludes BBBC036 plate 25503).")
    p.add_argument("--max-plates", type=int, default=None)
    p.add_argument("--max-fovs", type=int, default=None, help="Smoke: limit FOVs per plate.")
    p.add_argument("--overwrite", action="store_true")

    # DINOv2 backbone (mirrors inference.py)
    p.add_argument("--dinov2-repo", default="/home/gpznx/projects/DINOv2_CellPainting")
    p.add_argument("--ckpt", required=True, help="teacher_checkpoint.pth or merged *.full.pth.")
    p.add_argument("--config-file", default="", help="Training YAML (recommended, matches student cfg).")
    p.add_argument("--arch", default="vit_small")
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--in_chans", type=int, default=5)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--ffn_layer", default="mlp")
    p.add_argument("--block_chunks", type=int, default=1)
    p.add_argument("--num_register_tokens", type=int, default=0)
    p.add_argument("--strict-backbone", action="store_true")
    p.add_argument("opts", nargs=argparse.REMAINDER, default=[])

    # cropping / inference
    p.add_argument("--size", type=int, default=224, help="Crop size (model input).")
    p.add_argument("--stride", type=int, default=None, help="Crop stride (default = size).")
    p.add_argument("--operation", default="mean", choices=["mean", "median"])
    p.add_argument("--batch-size", type=int, default=8, help="FOVs per batch.")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--gpus", nargs="*", type=int, default=[0])
    return p


if __name__ == "__main__":
    run(get_parser().parse_args())
