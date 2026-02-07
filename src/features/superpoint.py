"""
SuperPoint feature detector implementation.
Based on "SuperPoint: Self-Supervised Interest Point Detection and Description"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SuperPointNet(nn.Module):
    """
    SuperPoint neural network architecture.
    """
    
    def __init__(self):
        super(SuperPointNet, self).__init__()
        
        # Shared encoder
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        
        # Encoder
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        
        # Detector head
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        
        # Descriptor head
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        """
        Forward pass.
        """
        # Shared encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        # Detector head
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        
        # Descriptor head
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        
        # Normalize descriptors
        dn = torch.norm(desc, p=2, dim=1, keepdim=True)
        desc = desc.div(dn + 1e-8)
        
        return semi, desc


class SuperPoint:
    """
    SuperPoint feature detector and descriptor.
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 nms_radius: int = 4,
                 keypoint_threshold: float = 0.005,
                 max_keypoints: int = 1024,
                 remove_borders: int = 4,
                 use_cuda: bool = True):
        """
        Initialize SuperPoint detector.
        """
        self.nms_radius = nms_radius
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.remove_borders = remove_borders
        
        # Setup device
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        # Initialize network
        self.net = SuperPointNet().to(self.device)
        self.net.eval()
        
        # Load pretrained weights if provided
        if model_path is not None and Path(model_path).exists():
            self.load_weights(model_path)
        else:
            if model_path is not None:
                logger.warning(f"Model file not found: {model_path}. Using random weights.")
            else:
                logger.warning("No model path provided. Using random weights.")
    
    def load_weights(self, model_path: str) -> None:
        """
        Load pretrained weights.
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            self.net.load_state_dict(state_dict)
            logger.info(f"Loaded SuperPoint weights from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            raise
    
    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for SuperPoint.
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image = image[:, :, 0]
        
        original_shape = image.shape[:2]
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return tensor, original_shape
    
    def _nms(self, scores: np.ndarray, radius: int) -> np.ndarray:
        """
        Apply non-maximum suppression.
        """
        # Pad scores
        pad = radius
        scores_pad = np.pad(scores, pad, mode='constant', constant_values=0)
        
        # Max pooling
        kernel_size = 2 * radius + 1
        max_pool = cv2.dilate(scores, np.ones((kernel_size, kernel_size)))
        
        # Keep only local maxima
        nms_mask = (scores == max_pool) & (scores > self.keypoint_threshold)
        
        return nms_mask
    
    def _sample_descriptors(self,
                           keypoints: np.ndarray,
                           descriptors: torch.Tensor,
                           h: int,
                           w: int) -> np.ndarray:
        """
        Sample descriptors at keypoint locations.
        """
        # Convert keypoints to descriptor map coordinates
        _, _, hc, wc = descriptors.shape
        keypoints_norm = keypoints.copy()
        keypoints_norm[:, 0] = keypoints_norm[:, 0] / (w / 2.0) - 1.0  # x
        keypoints_norm[:, 1] = keypoints_norm[:, 1] / (h / 2.0) - 1.0  # y
        
        # Convert to tensor
        keypoints_norm = torch.from_numpy(keypoints_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        # Sample descriptors using grid_sample
        sampled_desc = F.grid_sample(
            descriptors,
            keypoints_norm,
            mode='bilinear',
            align_corners=True
        )
        
        # Convert to numpy
        sampled_desc = sampled_desc.squeeze().t().cpu().numpy()
        
        return sampled_desc
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect keypoints and extract descriptors.
        """
        # Preprocess
        tensor, (h, w) = self._preprocess(image)
        
        # Forward pass
        with torch.no_grad():
            semi, coarse_desc = self.net(tensor)
        
        # Convert semi-dense heatmap to keypoint scores
        semi = F.softmax(semi, dim=1)
        
        # Remove dustbin (no keypoint) channel
        heatmap = semi[:, :-1, :, :]
        
        # Reshape to [B, H/8, W/8, 8, 8] and get max over 8x8 cells
        b, _, hc, wc = heatmap.shape
        heatmap = heatmap.permute(0, 2, 3, 1).reshape(b, hc, wc, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(b, hc * 8, wc * 8)
        
        # Convert to numpy
        heatmap = heatmap.squeeze().cpu().numpy()
        
        # Crop to original size
        heatmap = heatmap[:h, :w]
        
        # Apply NMS
        nms_mask = self._nms(heatmap, self.nms_radius)
        
        # Remove borders
        if self.remove_borders > 0:
            nms_mask[:self.remove_borders, :] = 0
            nms_mask[-self.remove_borders:, :] = 0
            nms_mask[:, :self.remove_borders] = 0
            nms_mask[:, -self.remove_borders:] = 0
        
        # Extract keypoints
        keypoints_yx = np.where(nms_mask)
        scores = heatmap[keypoints_yx]
        keypoints = np.stack([keypoints_yx[1], keypoints_yx[0]], axis=1)  # Convert to (x, y)
        
        # Sort by score and keep top-k
        if len(keypoints) > self.max_keypoints:
            indices = np.argsort(-scores)[:self.max_keypoints]
            keypoints = keypoints[indices]
            scores = scores[indices]
        
        # Sample descriptors at keypoint locations
        if len(keypoints) > 0:
            descriptors = self._sample_descriptors(keypoints, coarse_desc, h, w)
        else:
            descriptors = np.zeros((0, 256), dtype=np.float32)
        
        return keypoints.astype(np.float32), descriptors.astype(np.float32), scores.astype(np.float32)
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Callable interface for detection."""
        return self.detect(image)