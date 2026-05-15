from __future__ import annotations

from typing import Optional, Sequence, Union

import cv2
import numpy as np


class USBCapture:
    """Simple OpenCV-based RGB capture for USB webcams.

    This class mirrors the minimal interface expected by ``VideoCapture``:
    - ``name`` attribute
    - ``read() -> (ret: bool, frame: np.ndarray | None)``
    - ``close()``

    Typical example for a Linux UVC wrist cam:
    ``USBCapture(name="wrist", device="/dev/video0", dim=(640, 480), fps=30)``
    """

    def __init__(
        self,
        name: str,
        device: Union[int, str] = 2,
        dim: Sequence[int] = (640, 480),
        fps: int = 30,
        exposure: Optional[float] = None,
        backend: Optional[int] = None,
        warmup_reads: int = 5,
    ):
        self.name = name
        self.device = device

        # CAP_V4L2 is usually the most reliable backend for Linux USB UVC cams.
        if backend is None:
            backend = cv2.CAP_V4L2

        self.cap = cv2.VideoCapture(device, backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open USB camera '{name}' at device '{device}'. "
                "Set 'device' to the right /dev/videoX or index."
            )

        width, height = int(dim[0]), int(dim[1])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))

        # Many cheap USB2.0 cameras behave better with MJPG on Linux.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if exposure is not None:
            # Manual exposure semantics differ by backend/camera firmware.
            # This is a best-effort setting and can be tuned per device.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

        # Drop a few initial frames so auto controls can settle.
        for _ in range(max(0, int(warmup_reads))):
            self.cap.read()

    @staticmethod
    def list_video_devices(max_index: int = 10) -> list[int]:
        """Return camera indices that can be opened (quick probe helper)."""
        found = []
        for idx in range(int(max_index)):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                found.append(idx)
            cap.release()
        return found

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None
        return True, frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
