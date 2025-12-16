#!/usr/bin/env python3
"""
preprocess_video.py

Takes any video file and creates:
1. Downsampled 1080p 30fps version (test input)
2. Ground truth frames at original resolution/fps
3. Metadata and triplet index for evaluation

Usage:
    python scripts/preprocess_video.py --input video.mp4 --output data/processed/clip_001
    python scripts/preprocess_video.py --input video.mp4 --output data/processed/clip_001 --target-res 1440p
    python scripts/preprocess_video.py --input video.mp4 --output data/processed/clip_001 --target-fps 120

Example workflow:
    # Copy video from Windows to WSL
    cp /mnt/c/Users/Myko/Videos/tarkov_gameplay.mp4 ~/gaming-vfisr/data/raw/
    
    # Preprocess
    cd ~/gaming-vfisr
    source venv/bin/activate
    python scripts/preprocess_video.py \\
        --input data/raw/tarkov_gameplay.mp4 \\
        --output data/processed/tarkov_001 \\
        --target-res 1440p
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List



@dataclass
class VideoMetadata:
    """Metadata for a processed video clip"""
    source_path: str
    source_resolution: Tuple[int, int]
    source_fps: float
    source_duration: float
    source_frame_count: int
    
    input_resolution: Tuple[int, int]  # e.g., (1920, 1080)
    input_fps: float  # e.g., 30.0
    
    target_resolution: Tuple[int, int]  # e.g., (2560, 1440)
    target_fps: float  # e.g., 120.0
    
    temporal_scale: float  # e.g., 4.0 for 30->120
    spatial_scale: float   # e.g., 1.33 for 1080->1440
    
    num_input_frames: int
    num_gt_frames: int
    num_triplets: int
    
    processing_date: str
    

class VideoPreprocessor:
    """
    Preprocesses video for VFI+SR benchmarking.
    
    Pipeline:
    1. Analyze source video
    2. Create 1080p 30fps input (simulated low-quality capture)
    3. Extract ground truth at target resolution/fps
    4. Generate frame triplets for evaluation
    """
    
    # Standard resolutions
    RESOLUTIONS = {
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '1440p': (2560, 1440),
        '4k': (3840, 2160),
    }
    
    def __init__(
        self,
        input_path: str,
        output_dir: str,
        input_resolution: str = '1080p',
        input_fps: float = 30.0,
        target_resolution: Optional[str] = None,  # None = use source
        target_fps: Optional[float] = None,  # None = use source
        max_frames: Optional[int] = None,  # Limit for testing
    ):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.input_res = self.RESOLUTIONS.get(input_resolution, (1920, 1080))
        self.input_fps = input_fps
        self.target_res_name = target_resolution
        self.target_fps_override = target_fps
        self.max_frames = max_frames
        
        # Will be set during analysis
        self.source_info = None
        self.metadata = None
        
        # Validate input exists
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input video not found: {self.input_path}")
    
    def analyze_source(self) -> dict:
        """Analyze source video properties using ffprobe"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(self.input_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        
        probe = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in probe['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break
        
        if video_stream is None:
            raise ValueError("No video stream found in input file")
        
        # Parse framerate (can be "60/1" or "59.94")
        fps_str = video_stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 30.0
        else:
            fps = float(fps_str)
        
        # Get duration
        duration = float(probe['format'].get('duration', 0))
        
        # Get frame count
        frame_count = int(video_stream.get('nb_frames', 0))
        if frame_count == 0:
            # Estimate from duration
            frame_count = int(duration * fps)
        
        self.source_info = {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': fps,
            'duration': duration,
            'frame_count': frame_count,
            'codec': video_stream.get('codec_name', 'unknown'),
            'pix_fmt': video_stream.get('pix_fmt', 'unknown'),
        }
        
        return self.source_info
    
    def validate_source(self) -> List[str]:
        """Validate source video and return warnings"""
        warnings = []
        info = self.source_info
        
        # Check resolution
        if info['width'] < 1920 or info['height'] < 1080:
            warnings.append(
                f"Source resolution ({info['width']}x{info['height']}) is below 1080p. "
                "Ground truth quality will be limited."
            )
        
        # Check framerate for VFI testing
        if info['fps'] < 60:
            warnings.append(
                f"Source framerate ({info['fps']:.2f} fps) is below 60fps. "
                "For 4x VFI testing (30->120fps), recommend 120fps+ source. "
                "Will use source fps as ground truth target."
            )
        
        # Check duration
        if info['duration'] < 1.0:
            raise ValueError("Video too short (< 1 second)")
        
        if info['duration'] < 5.0:
            warnings.append("Video is short (< 5 seconds). Limited test data.")
        
        return warnings
    
    def setup_output_dirs(self) -> dict:
        """Create output directory structure"""
        dirs = {
            'root': self.output_dir,
            'input': self.output_dir / 'input_1080p30',
            'input_frames': self.output_dir / 'input_1080p30' / 'frames',
            'input_video': self.output_dir / 'input_1080p30',
            'ground_truth': self.output_dir / 'ground_truth',
            'gt_frames': self.output_dir / 'ground_truth' / 'frames',
            'gt_video': self.output_dir / 'ground_truth',
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return dirs
    
    def create_input_video(self, dirs: dict) -> Path:
        """
        Create 1080p 30fps input video using FFmpeg.
        This simulates the low-quality input that models will process.
        """
        output_path = dirs['input_video'] / 'input.mp4'
        
        # Build filter string
        vf_filters = [
            f'scale={self.input_res[0]}:{self.input_res[1]}:flags=lanczos'
        ]
        
        # Add frame limit if specified
        duration_args = []
        if self.max_frames:
            duration = self.max_frames / self.input_fps
            duration_args = ['-t', str(duration)]
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite
            '-i', str(self.input_path),
            *duration_args,
            '-vf', ','.join(vf_filters),
            '-r', str(self.input_fps),
            '-c:v', 'libx264',
            '-crf', '18',  # High quality (we want clean input, just lower res/fps)
            '-preset', 'slow',
            '-pix_fmt', 'yuv420p',
            '-an',  # No audio
            str(output_path)
        ]
        
        print(f"Creating input video: {self.input_res[0]}x{self.input_res[1]} @ {self.input_fps}fps")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError("Failed to create input video")
        
        return output_path
    
    def create_ground_truth_video(self, dirs: dict) -> Path:
        """
        Create ground truth video at target resolution/fps.
        """
        # Determine target resolution
        if self.target_res_name:
            target_res = self.RESOLUTIONS.get(self.target_res_name)
            if target_res is None:
                raise ValueError(f"Unknown resolution: {self.target_res_name}")
        else:
            target_res = (self.source_info['width'], self.source_info['height'])
        
        # Determine target fps
        if self.target_fps_override:
            target_fps = self.target_fps_override
        else:
            target_fps = self.source_info['fps']
        
        output_path = dirs['gt_video'] / 'ground_truth.mp4'
        
        # Build filter chain
        vf_filters = []
        
        # Scale if target differs from source
        if target_res != (self.source_info['width'], self.source_info['height']):
            vf_filters.append(f'scale={target_res[0]}:{target_res[1]}:flags=lanczos')
        
        # FPS change only if reducing (can't create frames that don't exist)
        if target_fps < self.source_info['fps']:
            vf_filters.append(f'fps={target_fps}')
        elif target_fps > self.source_info['fps']:
            print(f"  Warning: Target fps ({target_fps}) > source fps ({self.source_info['fps']:.2f})")
            print("           Using source fps as ground truth")
            target_fps = self.source_info['fps']
        
        # Add frame limit if specified
        duration_args = []
        if self.max_frames:
            # Calculate equivalent duration at target fps
            input_duration = self.max_frames / self.input_fps
            duration_args = ['-t', str(input_duration)]
        
        cmd = [
            'ffmpeg',
            '-y',
            '-i', str(self.input_path),
            *duration_args,
        ]
        
        if vf_filters:
            cmd.extend(['-vf', ','.join(vf_filters)])
        
        cmd.extend([
            '-c:v', 'libx264',
            '-crf', '15',  # Very high quality for ground truth
            '-preset', 'slow',
            '-pix_fmt', 'yuv420p',
            '-an',
            str(output_path)
        ])
        
        print(f"Creating ground truth: {target_res[0]}x{target_res[1]} @ {target_fps}fps")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError("Failed to create ground truth video")
        
        # Store for metadata
        self.target_res = target_res
        self.target_fps = target_fps
        
        return output_path
    
    def extract_frames(self, video_path: Path, output_dir: Path, prefix: str = 'frame') -> int:
        """Extract all frames from a video as PNG"""
        cmd = [
            'ffmpeg',
            '-y',
            '-i', str(video_path),
            '-vsync', '0',
            '-start_number', '0',
            str(output_dir / f'{prefix}_%06d.png')
        ]
        
        print(f"Extracting frames from {video_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"Failed to extract frames from {video_path}")
        
        # Count extracted frames
        frame_count = len(list(output_dir.glob(f'{prefix}_*.png')))
        return frame_count
    
    def create_triplet_index(self, dirs: dict) -> List[dict]:
        """
        Create index mapping input frames to ground truth frames.
        
        For 30fps -> 120fps (4x temporal scale):
        - Every input frame pair spans multiple GT frames
        - We need to generate 3 intermediate frames per pair
        
        Input:  [0] -------- [1] -------- [2]
        GT:     [0] [1] [2] [3] [4] [5] [6] [7] [8]
        
        Triplet 0: input[0], input[1] -> gt[1], gt[2], gt[3]
        Triplet 1: input[1], input[2] -> gt[5], gt[6], gt[7]
        """
        input_frames = sorted(dirs['input_frames'].glob('frame_*.png'))
        gt_frames = sorted(dirs['gt_frames'].glob('frame_*.png'))
        
        if not input_frames:
            raise ValueError("No input frames found")
        if not gt_frames:
            raise ValueError("No ground truth frames found")
        
        temporal_scale = self.target_fps / self.input_fps
        
        print(f"  Input frames: {len(input_frames)}")
        print(f"  GT frames: {len(gt_frames)}")
        print(f"  Temporal scale: {temporal_scale:.2f}x")
        
        triplets = []
        
        for i in range(len(input_frames) - 1):
            # Input frame indices
            input_idx_0 = i
            input_idx_1 = i + 1
            
            # Corresponding GT frame indices
            gt_start = int(round(i * temporal_scale))
            gt_end = int(round((i + 1) * temporal_scale))
            
            # Intermediate GT frames (excluding endpoints)
            gt_intermediate_indices = list(range(gt_start + 1, gt_end))
            
            # Skip if no intermediate frames or indices out of bounds
            if not gt_intermediate_indices:
                continue
            if gt_end >= len(gt_frames):
                continue
            
            # Calculate temporal positions for each intermediate frame
            temporal_positions = [
                (idx - gt_start) / (gt_end - gt_start)
                for idx in gt_intermediate_indices
            ]
            
            triplet = {
                'triplet_id': len(triplets),
                'input_frame_0': input_frames[input_idx_0].name,
                'input_frame_1': input_frames[input_idx_1].name,
                'input_idx_0': input_idx_0,
                'input_idx_1': input_idx_1,
                'gt_frame_start': gt_frames[gt_start].name,
                'gt_frame_end': gt_frames[gt_end].name,
                'gt_start_idx': gt_start,
                'gt_end_idx': gt_end,
                'gt_intermediate': [
                    gt_frames[idx].name
                    for idx in gt_intermediate_indices
                    if idx < len(gt_frames)
                ],
                'gt_intermediate_indices': [
                    idx for idx in gt_intermediate_indices if idx < len(gt_frames)
                ],
                'temporal_positions': temporal_positions,
                'num_intermediate': len(gt_intermediate_indices),
            }
            triplets.append(triplet)
        
        return triplets
    
    def generate_metadata(self, dirs: dict, triplets: List[dict]) -> VideoMetadata:
        """Generate metadata for the processed clip"""
        temporal_scale = self.target_fps / self.input_fps
        
        # Calculate spatial scale (linear, not area)
        spatial_scale = self.target_res[0] / self.input_res[0]
        
        input_frames = len(list(dirs['input_frames'].glob('frame_*.png')))
        gt_frames = len(list(dirs['gt_frames'].glob('frame_*.png')))
        
        self.metadata = VideoMetadata(
            source_path=str(self.input_path.absolute()),
            source_resolution=(self.source_info['width'], self.source_info['height']),
            source_fps=self.source_info['fps'],
            source_duration=self.source_info['duration'],
            source_frame_count=self.source_info['frame_count'],
            
            input_resolution=self.input_res,
            input_fps=self.input_fps,
            
            target_resolution=self.target_res,
            target_fps=self.target_fps,
            
            temporal_scale=temporal_scale,
            spatial_scale=spatial_scale,
            
            num_input_frames=input_frames,
            num_gt_frames=gt_frames,
            num_triplets=len(triplets),
            
            processing_date=datetime.now().isoformat(),
        )
        
        return self.metadata
    
    def save_outputs(self, dirs: dict, triplets: List[dict]) -> None:
        """Save metadata and triplet index to JSON files"""
        # Main metadata
        metadata_path = dirs['root'] / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(asdict(self.metadata), f, indent=2)
        
        # Triplet index
        triplets_path = dirs['root'] / 'triplets.json'
        with open(triplets_path, 'w') as f:
            json.dump(triplets, f, indent=2)
        
        print(f"  Saved metadata to {metadata_path}")
        print(f"  Saved {len(triplets)} triplets to {triplets_path}")
    
    def process(self) -> dict:
        """Main processing pipeline"""
        print("=" * 60)
        print(" VIDEO PREPROCESSING PIPELINE")
        print("=" * 60)
        print(f"Input: {self.input_path}")
        print(f"Output: {self.output_dir}")
        
        # 1. Analyze source
        print("\n[1/7] Analyzing source video...")
        self.analyze_source()
        print(f"  Resolution: {self.source_info['width']}x{self.source_info['height']}")
        print(f"  Framerate: {self.source_info['fps']:.2f} fps")
        print(f"  Duration: {self.source_info['duration']:.2f}s")
        print(f"  Frames: {self.source_info['frame_count']}")
        print(f"  Codec: {self.source_info['codec']}")
        
        # 2. Validate
        print("\n[2/7] Validating source...")
        warnings = self.validate_source()
        for w in warnings:
            print(f"  WARNING: {w}")
        if not warnings:
            print("  All checks passed")
        
        # 3. Setup directories
        print("\n[3/7] Setting up output directories...")
        dirs = self.setup_output_dirs()
        print(f"  Created: {dirs['root']}")
        
        # 4. Create input video (1080p 30fps)
        print("\n[4/7] Creating input video...")
        input_video = self.create_input_video(dirs)
        print(f"  Saved: {input_video}")
        
        # 5. Create ground truth video
        print("\n[5/7] Creating ground truth video...")
        gt_video = self.create_ground_truth_video(dirs)
        print(f"  Saved: {gt_video}")
        
        # 6. Extract frames
        print("\n[6/7] Extracting frames...")
        input_frame_count = self.extract_frames(input_video, dirs['input_frames'], 'frame')
        gt_frame_count = self.extract_frames(gt_video, dirs['gt_frames'], 'frame')
        print(f"  Input frames: {input_frame_count}")
        print(f"  GT frames: {gt_frame_count}")
        
        # 7. Create triplet index and metadata
        print("\n[7/7] Generating metadata and triplet index...")
        triplets = self.create_triplet_index(dirs)
        self.generate_metadata(dirs, triplets)
        self.save_outputs(dirs, triplets)
        
        # Summary
        print("\n" + "=" * 60)
        print(" PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print("")
        print(f"Input:  {self.metadata.input_resolution[0]}x{self.metadata.input_resolution[1]} "
              f"@ {self.metadata.input_fps}fps")
        print(f"Target: {self.metadata.target_resolution[0]}x{self.metadata.target_resolution[1]} "
              f"@ {self.metadata.target_fps}fps")
        print("")
        print(f"Temporal scale: {self.metadata.temporal_scale:.2f}x "
              f"({self.metadata.input_fps} -> {self.metadata.target_fps} fps)")
        print(f"Spatial scale:  {self.metadata.spatial_scale:.2f}x "
              f"({self.metadata.input_resolution[0]} -> {self.metadata.target_resolution[0]})")
        print("")
        print(f"Triplets for evaluation: {len(triplets)}")
        print(f"Intermediate frames per triplet: {triplets[0]['num_intermediate'] if triplets else 0}")
        
        return {
            'metadata': asdict(self.metadata),
            'triplets': triplets,
            'dirs': {k: str(v) for k, v in dirs.items()},
        }


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess video for VFI+SR benchmarking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (source resolution/fps as ground truth)
  python preprocess_video.py --input gameplay.mp4 --output data/processed/clip_001

  # Specify 1440p ground truth
  python preprocess_video.py --input gameplay.mp4 --output data/processed/clip_001 --target-res 1440p

  # Limit to first 100 frames (for testing)
  python preprocess_video.py --input gameplay.mp4 --output data/processed/clip_001 --max-frames 100
        """
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input video file path'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--input-resolution',
        default='1080p',
        choices=['720p', '1080p'],
        help='Resolution for input (downsampled) video (default: 1080p)'
    )
    parser.add_argument(
        '--input-fps',
        type=float,
        default=30.0,
        help='FPS for input (downsampled) video (default: 30)'
    )
    parser.add_argument(
        '--target-res', '--target-resolution',
        default=None,
        choices=['1080p', '1440p', '4k'],
        help='Target resolution for ground truth (default: source resolution)'
    )
    parser.add_argument(
        '--target-fps',
        type=float,
        default=None,
        help='Target FPS for ground truth (default: source FPS)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of input frames to process (for testing)'
    )
    
    args = parser.parse_args()
    
    try:
        preprocessor = VideoPreprocessor(
            input_path=args.input,
            output_dir=args.output,
            input_resolution=args.input_resolution,
            input_fps=args.input_fps,
            target_resolution=args.target_res,
            target_fps=args.target_fps,
            max_frames=args.max_frames,
        )
        
        preprocessor.process()
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
