#!/usr/bin/env python3
"""
Download pretrained models for SLAM system.
"""
import os
import sys
import logging
from pathlib import Path
import urllib.request
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model URLs and checksums
MODELS = {
    'superpoint_v1.pth': {
        'url': 'https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth',
        'md5': None,  # Add if known
        'description': 'SuperPoint pretrained weights'
    },
    # Add other models as needed
    # 'superglue_indoor.pth': {...},
    # 'netvlad_vgg16.pth': {...},
}


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """
    Download file from URL with progress bar.
    """
    try:
        logger.info(f"Downloading {description}...")
        logger.info(f"URL: {url}")
        logger.info(f"Saving to: {output_path}")
        
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = min(int(count * block_size * 100 / total_size), 100)
                sys.stdout.write(f"\rProgress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
        print()  # New line after progress
        
        logger.info(f"Successfully downloaded {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {description}: {e}")
        return False


def verify_md5(file_path: Path, expected_md5: str) -> bool:
    """
    Verify file MD5 checksum.
    """
    if expected_md5 is None:
        return True
    
    logger.info(f"Verifying MD5 checksum...")
    
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    actual_md5 = md5_hash.hexdigest()
    
    if actual_md5 == expected_md5:
        logger.info("Checksum verified ✓")
        return True
    else:
        logger.error(f"Checksum mismatch! Expected: {expected_md5}, Got: {actual_md5}")
        return False


def download_models(models_dir: Path = None, force: bool = False):
    """
    Download all pretrained models.
    """
    # Default models directory
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / 'models'
    
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Downloading pretrained models for SLAM system")
    logger.info("="*60)
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Number of models: {len(MODELS)}")
    logger.info("="*60 + "\n")
    
    success_count = 0
    
    for model_name, model_info in MODELS.items():
        output_path = models_dir / model_name
        
        # Check if already exists
        if output_path.exists() and not force:
            logger.info(f"✓ {model_name} already exists (skipping)")
            
            # Verify checksum if available
            if model_info['md5'] is not None:
                if verify_md5(output_path, model_info['md5']):
                    success_count += 1
                else:
                    logger.warning(f"Checksum failed for {model_name}, consider re-downloading")
            else:
                success_count += 1
            
            continue
        
        # Download
        if download_file(model_info['url'], output_path, model_info['description']):
            # Verify checksum
            if model_info['md5'] is not None:
                if verify_md5(output_path, model_info['md5']):
                    success_count += 1
                else:
                    logger.error(f"Removing corrupted file: {output_path}")
                    output_path.unlink()
            else:
                success_count += 1
        
        print()  # Separator
    
    logger.info("="*60)
    logger.info(f"Download complete: {success_count}/{len(MODELS)} models")
    logger.info("="*60)
    
    if success_count == len(MODELS):
        logger.info("All models downloaded successfully!")
        return 0
    else:
        logger.warning(f"{len(MODELS) - success_count} model(s) failed to download")
        return 1


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download pretrained models')
    parser.add_argument('--models-dir', type=str, default=None,
                       help='Directory to save models')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if files exist')
    args = parser.parse_args()
    
    return download_models(
        models_dir=args.models_dir,
        force=args.force
    )


if __name__ == '__main__':
    sys.exit(main())