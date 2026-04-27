#!/usr/bin/env python3
"""Run ForensicFace with ONNX-only backend and print full face extraction output.

This script intentionally prints all extracted values, including complete arrays
(embeddings, keypoints, bbox, aligned_face pixels, etc.) without min/max/mean
summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from forensicface.app import ForensicFace


def _to_plain(value: Any) -> Any:
    """Recursively convert numpy-heavy objects into JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sepaelv2"],
        help="One or more model pack names under ~/.forensicface/models",
    )
    parser.add_argument("--det-size", type=int, default=320, help="Detector input size")
    parser.add_argument(
        "--det-thresh", type=float, default=0.5, help="Detector confidence threshold"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument(
        "--multi-face",
        action="store_true",
        help="Return all detected faces instead of a single selected face",
    )
    parser.add_argument(
        "--draw-keypoints",
        action="store_true",
        help="Draw keypoints on aligned face image in output",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional file path to write full output JSON",
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
        extended=True,
        backend_name="onnx",
    )

    result = ff.process_image(
        str(image_path),
        single_face=not args.multi_face,
        draw_keypoints=args.draw_keypoints,
    )

    # Force complete array materialization in prints.
    np.set_printoptions(threshold=np.inf)
    full_output = _to_plain(result)

    serialized = json.dumps(full_output, indent=2, ensure_ascii=True)
    print(serialized)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"\nSaved full output JSON to: {output_path}")


if __name__ == "__main__":
    main()
