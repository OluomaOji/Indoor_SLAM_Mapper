#!/usr/bin/env python3
"""
Phase 1: Feature Detection Demo
Run real-time SuperPoint feature detection on camera feed.
"""
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import Camera, Frame, Config, get_config
from features import SuperPoint
from visualization import RealtimeDisplay

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for Phase 1 demo."""
    parser = argparse.ArgumentParser(description='SLAM Phase 1: Feature Detection')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--source', type=int, default=None,
                       help='Camera source (0 for webcam)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA')
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Override config with command line args
    if args.source is not None:
        config['camera.source'] = args.source
    if args.video is not None:
        config['camera.source'] = args.video
    if args.no_cuda:
        config['superpoint.use_cuda'] = False
    
    logger.info("="*60)
    logger.info("SLAM Phase 1: Feature Detection")
    logger.info("="*60)
    logger.info(f"Camera source: {config['camera.source']}")
    logger.info(f"Resolution: {config['camera.width']}x{config['camera.height']}")
    logger.info(f"Using CUDA: {config['superpoint.use_cuda']}")
    logger.info("="*60)
    
    # Initialize camera
    logger.info("Initializing camera...")
    camera = Camera(
        source=config['camera.source'],
        width=config['camera.width'],
        height=config['camera.height'],
        fps=config['camera.fps']
    )
    
    if not camera.open():
        logger.error("Failed to open camera!")
        return 1
    
    logger.info("Camera opened successfully")
    
    # Load calibration if available
    calib_file = Path(config['camera.calibration_file'])
    if calib_file.exists():
        logger.info(f"Loading calibration from {calib_file}")
        camera.load_calibration(calib_file)
    else:
        logger.warning("No calibration file found, using default intrinsics")
    
    # Initialize SuperPoint
    logger.info("Initializing SuperPoint feature detector...")
    detector = SuperPoint(
        model_path=config['superpoint.model_path'],
        nms_radius=config['superpoint.nms_radius'],
        keypoint_threshold=config['superpoint.keypoint_threshold'],
        max_keypoints=config['superpoint.max_keypoints'],
        remove_borders=config['superpoint.remove_borders'],
        use_cuda=config['superpoint.use_cuda']
    )
    logger.info("SuperPoint initialized")
    
    # Initialize display
    logger.info("Initializing display...")
    display = RealtimeDisplay(
        window_name=config['visualization.window_name'],
        show_fps=config['visualization.display_fps'],
        max_fps=config['visualization.max_fps']
    )
    logger.info("Display initialized")
    
    logger.info("\n" + "="*60)
    logger.info("Starting feature detection...")
    logger.info("Press 'q' or ESC to quit")
    logger.info("="*60 + "\n")
    
    frame_id = 0
    
    try:
        while True:
            # Read frame from camera
            ret, image, timestamp = camera.read()
            
            if not ret:
                logger.warning("Failed to read frame from camera")
                break
            
            # Create Frame object
            frame = Frame(
                frame_id=frame_id,
                timestamp=timestamp,
                image=image,
                K=camera.K
            )
            
            # Detect features
            keypoints, descriptors, scores = detector.detect(frame.gray)
            
            # Update frame with features
            frame.keypoints = keypoints
            frame.descriptors = descriptors
            frame.scores = scores
            
            # Display
            info = (f"Frame: {frame_id}\n"
                   f"Keypoints: {len(keypoints)}\n"
                   f"Resolution: {frame.width}x{frame.height}")
            
            if not display.show_features(
                image=frame.image,
                keypoints=frame.keypoints,
                scores=frame.scores,
                color=tuple(config['visualization.feature_color']),
                radius=config['visualization.feature_radius'],
                info=info
            ):
                break
            
            frame_id += 1
            
            # Log progress every 100 frames
            if frame_id % 100 == 0:
                logger.info(f"Processed {frame_id} frames")
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        camera.release()
        display.close()
        logger.info(f"Processed total of {frame_id} frames")
        logger.info("Done!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())