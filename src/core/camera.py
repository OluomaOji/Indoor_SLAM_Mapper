"""
Camera feed acquisition for SLAM system.
"""
import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class Camera:
    """
    Camera interface for acquiring frames from webcam or video file.
    """
    
    def __init__(self,
                 source: int | str = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 K: Optional[np.ndarray] = None,
                 distortion: Optional[np.ndarray] = None):
        """
        Initialize camera.
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        
        # Camera intrinsics
        self.K = K if K is not None else self._default_intrinsics()
        self.distortion = distortion if distortion is not None else np.zeros(5)
        
        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera = isinstance(source, int)
        self.is_open = False
        
        # Frame tracking
        self.frame_count = 0
        self.last_frame_time = 0.0
        
    def _default_intrinsics(self) -> np.ndarray:
        """
        Create default camera intrinsic matrix.
        Assumes focal length ~= width and principal point at center.
        """
        fx = fy = self.width
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    def open(self) -> bool:
        """
        Open camera/video source.
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera source: {self.source}")
                return False
            
            # Set camera properties
            if self.is_camera:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Read actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            
            # Update intrinsics if resolution changed
            if actual_width != self.width or actual_height != self.height:
                scale_x = actual_width / self.width
                scale_y = actual_height / self.height
                self.K[0, 0] *= scale_x  # fx
                self.K[1, 1] *= scale_y  # fy
                self.K[0, 2] *= scale_x  # cx
                self.K[1, 2] *= scale_y  # cy
                self.width = actual_width
                self.height = actual_height
            
            self.is_open = True
            return True
            
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Read a frame from the camera.
        """
        if not self.is_open or self.cap is None:
            return False, None, 0.0
        
        ret, frame = self.cap.read()
        
        if not ret:
            return False, None, 0.0
        
        # Generate timestamp
        timestamp = time.time()
        
        # Undistort if distortion coefficients are provided
        if np.any(self.distortion != 0):
            frame = cv2.undistort(frame, self.K, self.distortion)
        
        self.frame_count += 1
        self.last_frame_time = timestamp
        
        return True, frame, timestamp
    
    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_open = False
            logger.info("Camera released")
    
    def set_intrinsics(self, K: np.ndarray) -> None:
        """
        Set camera intrinsic matrix.
        """
        self.K = K.copy()
    
    def set_distortion(self, distortion: np.ndarray) -> None:
        """
        Set distortion coefficients.
        """
        self.distortion = distortion.copy()
    
    def load_calibration(self, calibration_file: str | Path) -> bool:
        """
        Load camera calibration from file.
        """
        import yaml
        
        try:
            with open(calibration_file, 'r') as f:
                calib = yaml.safe_load(f)
            
            # Load intrinsics
            if 'camera_matrix' in calib:
                self.K = np.array(calib['camera_matrix'], dtype=np.float32)
            
            # Load distortion
            if 'distortion_coefficients' in calib:
                self.distortion = np.array(calib['distortion_coefficients'], dtype=np.float32)
            
            logger.info(f"Loaded calibration from {calibration_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration: {e}")
            return False
    
    def get_fps(self) -> float:
        """
        Calculate actual FPS based on recent frames.
        """
        if self.cap is None or not self.is_open:
            return 0.0
        
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor."""
        self.release()


def load_calibration_from_yaml(filepath: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera calibration from YAML file.
    """
    import yaml
    
    with open(filepath, 'r') as f:
        calib = yaml.safe_load(f)
    
    K = np.array(calib.get('camera_matrix', np.eye(3)), dtype=np.float32)
    distortion = np.array(calib.get('distortion_coefficients', np.zeros(5)), dtype=np.float32)
    
    return K, distortion


def save_calibration_to_yaml(filepath: str | Path,
                             K: np.ndarray,
                             distortion: np.ndarray,
                             image_size: Tuple[int, int]) -> None:
    """
    Save camera calibration to YAML file.
    """
    import yaml
    
    calib_data = {
        'camera_matrix': K.tolist(),
        'distortion_coefficients': distortion.tolist(),
        'image_width': image_size[0],
        'image_height': image_size[1]
    }
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(calib_data, f, default_flow_style=False)
    
    logger.info(f"Saved calibration to {filepath}")

if __name__ == "__main__":
    # usage
    with Camera(source=0) as cam:
        while True:
            ret, frame, timestamp = cam.read()
            if not ret:
                break
            
            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()