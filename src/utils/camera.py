import cv2
import logging
from typing import Optional
import numpy as np

class CameraNotFoundError(Exception):
    pass

class CameraManager:
    def __init__(self, max_index: int = 10):
        self.max_index = max_index
        self.camera: Optional[cv2.VideoCapture] = None
        self.logger = logging.getLogger(__name__)

    def find_available_camera(self) -> cv2.VideoCapture:
        """Find the first available camera."""
        for i in range(self.max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.logger.info(f"Camera found at index {i}")
                return cap
        raise CameraNotFoundError("No available camera found")

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the camera."""
        if self.camera is None:
            self.logger.error("Camera not initialized")
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            self.logger.error("Failed to read frame")
            return None
        return frame

    def release(self) -> None:
        """Release the camera resources."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.logger.info("Camera released")

    def __enter__(self):
        """Context manager entry."""
        self.camera = self.find_available_camera()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release() 