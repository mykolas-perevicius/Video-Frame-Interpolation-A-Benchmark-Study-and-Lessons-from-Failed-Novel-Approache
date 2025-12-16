#!/usr/bin/env python3
"""
batch_preprocess.py

Process multiple video files at once.

Usage:
    python scripts/batch_preprocess.py --input-dir data/raw/ --output-dir data/processed/
    python scripts/batch_preprocess.py -i data/raw/ -o data/processed/ --target-res 1440p
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from preprocess_video import VideoPreprocessor


def find_video_files(input_dir: Path) -> List[Path]:
    """Find all video files in a directory"""
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v', '.flv'}
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f'*{ext}'))
        video_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    return sorted(video_files)


def batch_process(
    input_dir: str,
    output_dir: str,
    input_resolution: str = '1080p',
    input_fps: float = 30.0,
    target_resolution: str = None,
    target_fps: float = None,
    max_frames: int = None,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Process all video files in a directory.
    
    Args:
        input_dir: Directory containing video files
        output_dir: Output directory for processed data
        input_resolution: Resolution for downsampled input
        input_fps: FPS for downsampled input
        target_resolution: Target resolution for ground truth
        target_fps: Target FPS for ground truth
        max_frames: Maximum frames to process per video
        skip_existing: Skip videos that already have output directories
        
    Returns:
        Dictionary with processing results
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = find_video_files(input_path)
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return {'status': 'error', 'message': 'No video files found'}
    
    print("=" * 60)
    print(" BATCH VIDEO PREPROCESSING")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(video_files)} video files")
    print("=" * 60)
    
    results = []
    successful = 0
    skipped = 0
    failed = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
        print("-" * 40)
        
        # Generate output directory name
        clip_name = video_file.stem
        # Sanitize name (remove special characters)
        clip_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in clip_name)
        clip_output = output_path / clip_name
        
        # Check if already processed
        if skip_existing and clip_output.exists() and (clip_output / 'metadata.json').exists():
            print("  Skipping (already processed)")
            skipped += 1
            results.append({
                'file': str(video_file),
                'output': str(clip_output),
                'status': 'skipped',
            })
            continue
        
        try:
            preprocessor = VideoPreprocessor(
                input_path=str(video_file),
                output_dir=str(clip_output),
                input_resolution=input_resolution,
                input_fps=input_fps,
                target_resolution=target_resolution,
                target_fps=target_fps,
                max_frames=max_frames,
            )
            result = preprocessor.process()
            
            successful += 1
            results.append({
                'file': str(video_file),
                'output': str(clip_output),
                'status': 'success',
                'metadata': result.get('metadata', {}),
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
            results.append({
                'file': str(video_file),
                'output': str(clip_output),
                'status': 'error',
                'error': str(e),
            })
    
    # Summary
    print("\n" + "=" * 60)
    print(" BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total videos: {len(video_files)}")
    print(f"  Successful: {successful}")
    print(f"  Skipped:    {skipped}")
    print(f"  Failed:     {failed}")
    
    # Save batch results
    batch_results = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_path.absolute()),
        'output_dir': str(output_path.absolute()),
        'settings': {
            'input_resolution': input_resolution,
            'input_fps': input_fps,
            'target_resolution': target_resolution,
            'target_fps': target_fps,
            'max_frames': max_frames,
        },
        'summary': {
            'total': len(video_files),
            'successful': successful,
            'skipped': skipped,
            'failed': failed,
        },
        'results': results,
    }
    
    batch_log_path = output_path / 'batch_processing_log.json'
    with open(batch_log_path, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"\nBatch log saved to: {batch_log_path}")
    
    return batch_results


def main():
    parser = argparse.ArgumentParser(
        description='Batch preprocess videos for VFI+SR benchmarking'
    )
    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Directory containing video files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--input-resolution',
        default='1080p',
        choices=['720p', '1080p'],
        help='Resolution for downsampled input (default: 1080p)'
    )
    parser.add_argument(
        '--input-fps',
        type=float,
        default=30.0,
        help='FPS for downsampled input (default: 30)'
    )
    parser.add_argument(
        '--target-res',
        dest='target_resolution',
        default=None,
        choices=['1080p', '1440p', '4k'],
        help='Target resolution for ground truth (default: source)'
    )
    parser.add_argument(
        '--target-fps',
        type=float,
        default=None,
        help='Target FPS for ground truth (default: source)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames to process per video'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess videos even if output exists'
    )
    
    args = parser.parse_args()
    
    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_resolution=args.input_resolution,
        input_fps=args.input_fps,
        target_resolution=args.target_resolution,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        skip_existing=not args.force,
    )


if __name__ == '__main__':
    main()
