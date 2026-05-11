"""
Recorder — saves processed frames to .mp4
"""

import cv2
import os
from datetime import datetime


class Recorder:
    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = output_dir
        self._writer    = None

    def start(self, frame_shape, fps: float = 30.0):
        os.makedirs(self.output_dir, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"session_{ts}.mp4")
        h, w = frame_shape[:2]
        fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        print(f"[REC] Writing to {path}")

    def write(self, frame):
        if self._writer:
            self._writer.write(frame)

    def stop(self):
        if self._writer:
            self._writer.release()
            self._writer = None
