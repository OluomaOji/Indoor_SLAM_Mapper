#!/usr/bin/env python3
"""
Camera calibration using checkerboard pattern.
"""
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import Camera, save_calibration_to_yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calibrate_camera(
    camera_source: int = 0,
    checkerboard_size: tuple = (9, 6),
    square_size: float = 0.025,  # 25mm squares
    num_images: int = 20,
    output_file: str = None
):
    """
    Calibrate camera using checkerboard pattern.
    """
    logger.info("="*60)
    logger.info("Camera Calibration")
    logger.info("="*60)
    logger.info(f"Checkerboard size: {checkerboard_size}")
    logger.info(f"Square size: {square_size*1000:.1f}mm")
    logger.info(f"Target images: {num_images}")
    logger.info("="*60)
    logger.info("\nInstructions:")
    logger.info("1. Print a checkerboard pattern")
    logger.info("2. Hold it in front of the camera")
    logger.info("3. Press SPACE to capture when corners are detected")
    logger.info("4. Move the checkerboard to different positions/angles")
    logger.info("5. Press 'q' when done or after capturing enough images")
    logger.info("="*60 + "\n")
    
    # Prepare object points (3D points in real world space)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Open camera
    camera = Camera(source=camera_source)
    if not camera.open():
        logger.error("Failed to open camera")
        return None
    
    # Create window
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 1280, 720)
    
    captured_count = 0
    image_size = None
    
    try:
        while captured_count < num_images:
            # Read frame
            ret, frame, _ = camera.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(
                gray,
                checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            # Display frame
            display = frame.copy()
            
            if ret:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Draw corners
                cv2.drawChessboardCorners(display, checkerboard_size, corners_refined, ret)
                
                # Show capture prompt
                cv2.putText(
                    display,
                    "Press SPACE to capture",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    display,
                    "Checkerboard not detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
            # Show progress
            cv2.putText(
                display,
                f"Captured: {captured_count}/{num_images}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.imshow("Calibration", display)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and ret:
                # Capture image
                objpoints.append(objp)
                imgpoints.append(corners_refined)
                captured_count += 1
                logger.info(f"Captured image {captured_count}/{num_images}")
                
            elif key == ord('q'):
                break
        
        cv2.destroyWindow("Calibration")
        
        if captured_count < 3:
            logger.error("Not enough calibration images captured (need at least 3)")
            return None
        
        logger.info(f"\nCalibrating with {captured_count} images...")
        
        # Calibrate camera
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            None,
            None
        )
        
        if not ret:
            logger.error("Calibration failed")
            return None
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        
        logger.info("\n" + "="*60)
        logger.info("Calibration Results")
        logger.info("="*60)
        logger.info(f"RMS reprojection error: {mean_error:.4f} pixels")
        logger.info(f"\nCamera Matrix (K):")
        logger.info(f"{K}")
        logger.info(f"\nDistortion Coefficients:")
        logger.info(f"{dist.ravel()}")
        logger.info("="*60)
        
        # Save calibration
        if output_file is None:
            output_file = Path(__file__).parent.parent / "data" / "calibration" / "camera_params.yaml"
        
        save_calibration_to_yaml(output_file, K, dist.ravel(), image_size)
        logger.info(f"\nCalibration saved to: {output_file}")
        
        return K, dist.ravel(), mean_error
        
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Camera calibration using checkerboard')
    parser.add_argument('--source', type=int, default=0,
                       help='Camera source (default: 0)')
    parser.add_argument('--checkerboard', type=str, default='9,6',
                       help='Checkerboard size as "cols,rows" (default: 9,6)')
    parser.add_argument('--square-size', type=float, default=0.025,
                       help='Square size in meters (default: 0.025 = 25mm)')
    parser.add_argument('--num-images', type=int, default=20,
                       help='Number of calibration images (default: 20)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output calibration file')
    args = parser.parse_args()
    
    # Parse checkerboard size
    checkerboard_size = tuple(map(int, args.checkerboard.split(',')))
    
    result = calibrate_camera(
        camera_source=args.source,
        checkerboard_size=checkerboard_size,
        square_size=args.square_size,
        num_images=args.num_images,
        output_file=args.output
    )
    
    return 0 if result is not None else 1


if __name__ == '__main__':
    sys.exit(main())