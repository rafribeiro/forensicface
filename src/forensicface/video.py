"""Video face extraction workflow."""

from __future__ import annotations

import os

import cv2
from tqdm import tqdm

from .geometry import extend_bbox


__all__ = ["extract_faces_from_video"]


def extract_faces_from_video(
    processor,
    video_path: str,
    dest_folder: str = None,
    every_n_frames: int = 1,
    margin: float = 2.0,
    start_from: float = 0.0,
    export_metadata: bool = False,
) -> int:
    """Extract detected face crops from a video using a processor object."""
    import pandas as pd

    if dest_folder is None:
        dest_folder = os.path.splitext(video_path)[0]

    os.makedirs(dest_folder, exist_ok=True)

    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    start_frame = int(fps * start_from)
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT)) // every_n_frames

    vs.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    nfaces = 0
    if export_metadata:
        metadata = []
    with tqdm(
        total=total_frames,
        bar_format="Frames processed: {n}/{total} | Time elapsed: {elapsed}",
    ) as pbar:
        while True:
            ret, frame = vs.read()

            if not ret:
                break

            current_frame = current_frame + 1
            if (current_frame % every_n_frames) != 0:
                continue

            rets = processor.process_image(frame, single_face=False)
            for i, ret in enumerate(rets):
                out_bbox = extend_bbox(
                    ret["bbox"], frame.shape, margin_factor=margin
                )
                face_crop = frame[
                    out_bbox[1] : out_bbox[3], out_bbox[0] : out_bbox[2]
                ]
                face_img_path = os.path.join(
                    dest_folder, f"frame_{current_frame:07}_face_{i:02}.png"
                )
                cv2.imwrite(face_img_path, face_crop)
                if export_metadata:
                    metadata.append(
                        {
                            **{"frame": current_frame, "face": i},
                            **{
                                k: v
                                for k, v in ret.items()
                                if k not in ["det_score", "aligned_face"]
                            },
                        }
                    )
                nfaces += 1
            pbar.update(1)
    vs.release()
    if export_metadata:
        pd.DataFrame(metadata).to_json(
            os.path.join(
                dest_folder,
                os.path.splitext(os.path.basename(video_path))[0] + ".jsonl",
            ),
            lines=True,
            orient="records",
        )
    return nfaces
