#!/usr/bin/env python3
"""GPU preflight checks for the distinct DINOv2 448x448 model path."""

import argparse
import logging
import os

import torch

import config


def main():
    parser = argparse.ArgumentParser(description="Validate DINOv2 448x448 GPU inference shapes")
    parser.add_argument("--config", required=True, help="Inference config that registers dino_v2_448")
    parser.add_argument("--model", default="dino_v2_448", help="Model key to validate")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy validation batch size")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [dino448-preflight] %(levelname)s: %(message)s")
    logger = logging.getLogger("dino448-preflight")

    cfg = config.load_config(args.config)
    if args.model != "dino_v2_448":
        raise ValueError(f"This preflight is only intended for dino_v2_448, got {args.model}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DINOv2 448 preflight in this environment")

    transform = (
        config.get_direct_transform(args.model, cfg)
        if os.path.basename(args.config).startswith("direct_")
        else config.get_transform(args.model, cfg)
    )
    bit_depth = cfg["dataset"]["bit_depth"]
    if bit_depth == 16:
        sample = torch.randint(0, 65535, (1, 696, 520), dtype=torch.int32)
    else:
        sample = torch.randint(0, 255, (1, 1080, 1080), dtype=torch.uint8)
    transformed = transform(sample)
    if tuple(transformed.shape) != (3, 448, 448):
        raise AssertionError(f"Expected transformed sample shape (3, 448, 448), got {tuple(transformed.shape)}")
    logger.info("Transform shape OK: %s", tuple(transformed.shape))

    device = torch.device("cuda:0")
    wrapper_cls = config.get_model_wrapper(args.model)
    model = wrapper_cls(
        dataset=[],
        preprocess_transform=None,
        dataloader_settings={"batch_size": args.batch_size},
        save_path="/tmp/dino_v2_448_preflight",
        logger=logger,
        device=device,
        use_mixed_precision=True,
    )

    images = transformed.unsqueeze(0).repeat(args.batch_size, 1, 1, 1).to(device)
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        output = model.infer(images)

    cls = output["x_norm_clstoken"]
    patch = output["x_norm_patchtokens"]
    expected_cls = (args.batch_size, 768)
    expected_patch = (args.batch_size, 1024, 768)
    if tuple(cls.shape) != expected_cls:
        raise AssertionError(f"Expected cls-token shape {expected_cls}, got {tuple(cls.shape)}")
    if tuple(patch.shape) != expected_patch:
        raise AssertionError(f"Expected patch-token shape {expected_patch}, got {tuple(patch.shape)}")
    logger.info("Internal DINOv2 shapes OK: cls=%s patch=%s", tuple(cls.shape), tuple(patch.shape))

    data_input = {
        "filename": [f"dummy_{i}.png" for i in range(args.batch_size)],
        "plate_name": ["dummy_plate"] * args.batch_size,
    }
    result = model.postprocess(data_input, output)
    for key in ("cls_token_features", "patch_token_features"):
        if tuple(result[key].shape) != expected_cls:
            raise AssertionError(f"Expected saved {key} shape {expected_cls}, got {tuple(result[key].shape)}")
    logger.info("Saved feature shapes OK: cls_token_features=%s patch_token_features=%s", result["cls_token_features"].shape, result["patch_token_features"].shape)
    logger.info("DINOv2 448 preflight passed")


if __name__ == "__main__":
    main()

