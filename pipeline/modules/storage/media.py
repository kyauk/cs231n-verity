"""Media helpers: JPEG frame decode + H.264 MP4 encode via ffmpeg.

Small, dependency-light utilities used by the ingestion drivers to turn camera
JPEG frames into MP4 clips. Kept here so the drivers depend on a module rather
than a loose script.
"""

from __future__ import annotations

import subprocess

import cv2
import numpy as np

DEFAULT_FPS = 10


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode one JPEG-encoded camera frame into a BGR array."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG frame")
    return img


def encode_mp4_ffmpeg(frames: list[np.ndarray], out_path: str, fps: int = DEFAULT_FPS) -> None:
    """Encode a list of BGR frames into an H.264 MP4 via ffmpeg (libx264, yuv420p).

    All frames must share the first frame's dimensions, which must be even
    (yuv420p requirement) — crop to even before calling if needed.
    """
    if not frames:
        raise ValueError("No frames to encode")
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-crf", "23",
            out_path,
        ],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")
