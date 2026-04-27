#!/usr/bin/env python3
"""Run ForensicFace.process_image on a given image and print readable outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from forensicface.app import ForensicFace


def _to_readable(value: Any) -> Any:
    """Convert numpy-heavy objects into terminal-friendly summaries."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "size": 0,
            }
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "mean": float(np.mean(value)),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_readable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_readable(v) for v in value]
    return value


def _save_aligned_faces(result: dict[str, Any] | list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    results = result if isinstance(result, list) else [result]
    for idx, item in enumerate(results):
        aligned = item.get("aligned_face")
        if isinstance(aligned, np.ndarray):
            out_path = out_dir / f"aligned_face_{idx:02d}.png"
            bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), bgr)
            print(f"Saved aligned face: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sepaelv2"],
        help="One or more InsightFace model names",
    )
    parser.add_argument("--det-size", type=int, default=320, help="Face detector size")
    parser.add_argument("--det-thresh", type=float, default=0.5, help="Face detector threshold")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument(
        "--multi-face",
        action="store_true",
        help="Return all detected faces (default is single best face)",
    )
    parser.add_argument(
        "--draw-keypoints",
        action="store_true",
        help="Draw keypoints on aligned face output",
    )
    parser.add_argument(
        "--save-aligned-dir",
        default=None,
        help="Optional output directory to save aligned face image(s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ff = ForensicFace(
        models=args.models,
        det_size=args.det_size,
        use_gpu=not args.cpu,
        gpu=args.gpu,
        det_thresh=args.det_thresh,
    )

    result = ff.process_image(
        str(image_path),
        single_face=not args.multi_face,
        draw_keypoints=args.draw_keypoints,
    )

    print("=== process_image() readable output ===")
    print(json.dumps(_to_readable(result), indent=2, ensure_ascii=True))

    if args.save_aligned_dir:
        if isinstance(result, list) and len(result) == 0:
            print("No faces found; nothing to save.")
        else:
            _save_aligned_faces(result, Path(args.save_aligned_dir))


if __name__ == "__main__":
    main()
