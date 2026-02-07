"""
Frame data structure for SLAM system.
"""
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field
import cv2


@dataclass
class Frame:
    """
    Represents a single frame in the SLAM system.
    
    Stores image data, features, and associated metadata.
    """
    
    # Frame identification
    frame_id: int
    timestamp: float
    
    # Image data
    image: np.ndarray  # Original color image (BGR)
    gray: Optional[np.ndarray] = None  # Grayscale image
    
    # Camera intrinsics
    K: Optional[np.ndarray] = None  # 3x3 camera intrinsic matrix
    
    # Features (populated by feature detector)
    keypoints: Optional[np.ndarray] = None  # Nx2 array of keypoint locations
    descriptors: Optional[np.ndarray] = None  # NxD array of feature descriptors
    scores: Optional[np.ndarray] = None  # N array of keypoint scores
    
    # Pose information (populated by pose estimator)
    pose: Optional[np.ndarray] = None  # 4x4 transformation matrix (world to camera)
    
    # Metadata
    is_keyframe: bool = False
    
    def __post_init__(self):
        """Convert image to grayscale if not provided."""
        if self.gray is None and self.image is not None:
            if len(self.image.shape) == 3:
                self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                self.gray = self.image.copy()
    
    @property
    def height(self) -> int:
        """Get frame height."""
        return self.image.shape[0]
    
    @property
    def width(self) -> int:
        """Get frame width."""
        return self.image.shape[1]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get frame shape (height, width)."""
        return (self.height, self.width)
    
    @property
    def num_keypoints(self) -> int:
        """Get number of detected keypoints."""
        if self.keypoints is None:
            return 0
        return len(self.keypoints)
    
    def has_features(self) -> bool:
        """Check if frame has detected features."""
        return (self.keypoints is not None and 
                self.descriptors is not None and 
                len(self.keypoints) > 0)
    
    def has_pose(self) -> bool:
        """Check if frame has estimated pose."""
        return self.pose is not None
    
    def get_keypoints_cv2(self) -> list:
        """
        Convert keypoints to OpenCV KeyPoint format for visualization.
        """
        if self.keypoints is None:
            return []
        
        cv_keypoints = []
        for i, kp in enumerate(self.keypoints):
            size = float(self.scores[i]) * 10 if self.scores is not None else 1.0
            cv_kp = cv2.KeyPoint(
                x=float(kp[0]),
                y=float(kp[1]),
                size=size
            )
            cv_keypoints.append(cv_kp)
        
        return cv_keypoints
    
    def draw_keypoints(self, 
                       image: Optional[np.ndarray] = None,
                       color: Tuple[int, int, int] = (0, 255, 0),
                       radius: int = 3) -> np.ndarray:
        """
        Draw keypoints on image.
        """
        if image is None:
            image = self.image.copy()
        else:
            image = image.copy()
        
        if self.keypoints is None:
            return image
        
        for kp in self.keypoints:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), radius, color, -1)
        
        return image
    
    def clone(self) -> 'Frame':
        """Create a deep copy of the frame."""
        return Frame(
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            image=self.image.copy(),
            gray=self.gray.copy() if self.gray is not None else None,
            K=self.K.copy() if self.K is not None else None,
            keypoints=self.keypoints.copy() if self.keypoints is not None else None,
            descriptors=self.descriptors.copy() if self.descriptors is not None else None,
            scores=self.scores.copy() if self.scores is not None else None,
            pose=self.pose.copy() if self.pose is not None else None,
            is_keyframe=self.is_keyframe
        )
    
    def __repr__(self) -> str:
        """String representation of frame."""
        return (f"Frame(id={self.frame_id}, "
                f"shape={self.shape}, "
                f"keypoints={self.num_keypoints}, "
                f"keyframe={self.is_keyframe})")


class FrameBuffer:
    """
    Circular buffer for storing recent frames.
    """
    
    def __init__(self, max_size: int = 30):
        """
        Initialize frame buffer.
        """
        self.max_size = max_size
        self.frames = []
        self.frame_dict = {}  # frame_id -> Frame mapping
    
    def add(self, frame: Frame) -> None:
        """
        Add frame to buffer.
        """
        self.frames.append(frame)
        self.frame_dict[frame.frame_id] = frame
        
        # Remove oldest frame if buffer is full
        if len(self.frames) > self.max_size:
            old_frame = self.frames.pop(0)
            del self.frame_dict[old_frame.frame_id]
    
    def get_latest(self) -> Optional[Frame]:
        """Get most recent frame."""
        return self.frames[-1] if self.frames else None
    
    def get_by_id(self, frame_id: int) -> Optional[Frame]:
        """Get frame by ID."""
        return self.frame_dict.get(frame_id)
    
    def get_last_n(self, n: int) -> list:
        """Get last n frames."""
        return self.frames[-n:] if n <= len(self.frames) else self.frames
    
    def clear(self) -> None:
        """Clear all frames from buffer."""
        self.frames.clear()
        self.frame_dict.clear()
    
    def __len__(self) -> int:
        """Get number of frames in buffer."""
        return len(self.frames)
    
    def __iter__(self):
        """Iterate over frames."""
        return iter(self.frames)
