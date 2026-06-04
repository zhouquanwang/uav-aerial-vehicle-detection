"""Quick checks: model load + optional short sliced inference on a synthetic clip."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.class_labels import parse_class_labels_zh
from src.device_util import device_status_line, resolve_device
from src.predict_video import run_inference

MODEL = PROJECT_ROOT / "models" / "WALDO30_yolov8m_640x640 (1).pt"


def main() -> None:
    if not MODEL.is_file():
        raise SystemExit(f"Missing model: {MODEL}")

    print("Loading WALDO weights...")
    model = YOLO(str(MODEL))
    print("Classes:", model.names)
    print(device_status_line(resolve_device("auto")))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video = tmp_path / "smoke.mp4"
        w, h, fps, n = 1280, 720, 10.0, 3
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        for i in range(n):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"smoke frame {i}",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (200, 200, 200),
                2,
            )
            writer.write(frame)
        writer.release()

        out = tmp_path / "smoke_out.mp4"
        print("Running sliced inference (3 frames)...")
        frames = run_inference(
            model_path=MODEL,
            input_video=video,
            output_video=out,
            device=resolve_device("auto"),
            confidence_threshold=0.2,
            target_classes=[0],
            class_labels_zh=parse_class_labels_zh(None),
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
            detect_skip_interval=0,
            max_frames=3,
        )
        if frames != 3 or not out.is_file():
            raise SystemExit("Smoke inference failed")
        print(f"OK: wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
