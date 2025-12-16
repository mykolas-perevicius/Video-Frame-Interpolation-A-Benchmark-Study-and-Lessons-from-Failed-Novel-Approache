#!/usr/bin/env python3
"""
prepare_blind_study.py

Generate side-by-side comparison videos for blind user study.

Usage:
    python scripts/prepare_blind_study.py \\
        --results-dir outputs/model_outputs \\
        --ground-truth-dir data/processed \\
        --output-dir outputs/blind_study \\
        --models adaptive,rife,safa
"""

import argparse
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class ComparisonEntry:
    """Entry for blind study comparison"""
    comparison_id: str
    model_name: str
    clip_name: str
    order: str  # 'model_left' or 'model_right'
    video_path: str


def create_side_by_side_video(
    video_a_path: Path,
    video_b_path: Path,
    output_path: Path,
    label_a: str = "A",
    label_b: str = "B",
    font_scale: float = 1.5,
) -> bool:
    """
    Create side-by-side comparison video using FFmpeg.
    
    Returns True if successful.
    """
    # FFmpeg filter for side-by-side with labels
    filter_complex = (
        f"[0:v]drawtext=text='{label_a}':fontsize=48:fontcolor=white:"
        f"x=50:y=50:box=1:boxcolor=black@0.5:boxborderw=5[left];"
        f"[1:v]drawtext=text='{label_b}':fontsize=48:fontcolor=white:"
        f"x=50:y=50:box=1:boxcolor=black@0.5:boxborderw=5[right];"
        f"[left][right]hstack=inputs=2[out]"
    )
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_a_path),
        '-i', str(video_b_path),
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'medium',
        str(output_path),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  Timeout creating {output_path.name}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def create_comparison_from_frames(
    model_frames_dir: Path,
    gt_frames_dir: Path,
    output_path: Path,
    fps: float = 30.0,
    randomize_position: bool = True,
) -> Optional[str]:
    """
    Create comparison video from frame directories.
    
    Returns 'model_left' or 'model_right' indicating position.
    """
    # Get frame lists
    model_frames = sorted(model_frames_dir.glob('*.png'))
    gt_frames = sorted(gt_frames_dir.glob('*.png'))
    
    if not model_frames or not gt_frames:
        return None
    
    # Use minimum length
    num_frames = min(len(model_frames), len(gt_frames))
    
    # Determine order
    if randomize_position and random.random() > 0.5:
        left_frames, right_frames = model_frames[:num_frames], gt_frames[:num_frames]
        order = 'model_left'
    else:
        left_frames, right_frames = gt_frames[:num_frames], model_frames[:num_frames]
        order = 'model_right'
    
    # Read first frame to get dimensions
    sample = cv2.imread(str(left_frames[0]))
    h, w = sample.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w * 2, h))
    
    for left_path, right_path in zip(left_frames, right_frames):
        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        
        # Ensure same size
        if left.shape != right.shape:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))
        
        # Add labels
        cv2.putText(left, 'A', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(right, 'B', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Concatenate horizontally
        combined = np.hstack([left, right])
        out.write(combined)
    
    out.release()
    
    # Convert to proper codec with FFmpeg
    temp_path = output_path.with_suffix('.temp.mp4')
    output_path.rename(temp_path)
    
    cmd = [
        'ffmpeg', '-y',
        '-i', str(temp_path),
        '-c:v', 'libx264',
        '-crf', '18',
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True)
    temp_path.unlink()
    
    return order


def generate_participant_form(comparisons: List[ComparisonEntry], output_path: Path):
    """Generate markdown form for participants"""
    
    lines = [
        "# Video Quality Comparison Study",
        "",
        "## Instructions",
        "",
        "For each video pair, watch both sides (A and B) carefully.",
        "Then indicate which side you prefer based on:",
        "- **Sharpness**: Which is clearer/more detailed?",
        "- **Motion smoothness**: Which has smoother motion?",
        "- **Artifacts**: Which has fewer visual glitches?",
        "- **Overall preference**: Which looks better overall?",
        "",
        "You may watch each video multiple times.",
        "",
        "---",
        "",
    ]
    
    for i, entry in enumerate(comparisons, 1):
        lines.extend([
            f"## Comparison {i}: `{entry.comparison_id}`",
            "",
            f"Video file: `{Path(entry.video_path).name}`",
            "",
            "### Which side has better quality?",
            "",
            "- [ ] **A** is better",
            "- [ ] **B** is better",
            "- [ ] **Equal** (no preference)",
            "",
            "### Confidence (1-5): ___",
            "",
            "### Notes (optional):",
            "",
            "_________________________________________",
            "",
            "---",
            "",
        ])
    
    lines.extend([
        "## Participant Information",
        "",
        "- Participant ID: _______________",
        "- Display used: _______________",
        "- Viewing distance: _______________",
        "- Date: _______________",
        "",
        "Thank you for participating!",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def generate_answer_key(comparisons: List[ComparisonEntry], output_path: Path):
    """Generate answer key (SECRET - for unblinding)"""
    
    key_data = []
    for entry in comparisons:
        key_data.append({
            'comparison_id': entry.comparison_id,
            'model': entry.model_name,
            'clip': entry.clip_name,
            'order': entry.order,
            'model_is': 'A' if entry.order == 'model_left' else 'B',
            'ground_truth_is': 'B' if entry.order == 'model_left' else 'A',
        })
    
    with open(output_path, 'w') as f:
        json.dump(key_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare blind study comparison videos'
    )
    parser.add_argument(
        '--results-dir', '-r',
        required=True,
        help='Directory containing model output frames/videos'
    )
    parser.add_argument(
        '--ground-truth-dir', '-g',
        required=True,
        help='Directory containing ground truth (preprocessed clips)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Output directory for blind study materials'
    )
    parser.add_argument(
        '--models', '-m',
        default='adaptive,rife',
        help='Comma-separated list of models to compare'
    )
    parser.add_argument(
        '--max-comparisons',
        type=int,
        default=20,
        help='Maximum number of comparisons to generate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    results_dir = Path(args.results_dir)
    gt_dir = Path(args.ground_truth_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_compare = [m.strip() for m in args.models.split(',')]
    
    print("=" * 60)
    print(" BLIND STUDY PREPARATION")
    print("=" * 60)
    print(f"Models: {models_to_compare}")
    print(f"Output: {output_dir}")
    print()
    
    comparisons = []
    
    # Find available clips
    clips = [d.name for d in gt_dir.iterdir() if d.is_dir() and (d / 'metadata.json').exists()]
    print(f"Found {len(clips)} clips")
    
    for model_name in models_to_compare:
        print(f"\nProcessing model: {model_name}")
        
        model_output_dir = results_dir / model_name
        if not model_output_dir.exists():
            print(f"  Warning: No outputs found for {model_name}")
            continue
        
        for clip_name in clips:
            if len(comparisons) >= args.max_comparisons:
                break
            
            # Look for model outputs
            model_clip_dir = model_output_dir / clip_name
            if not model_clip_dir.exists():
                continue
            
            # Ground truth
            gt_clip_dir = gt_dir / clip_name / 'ground_truth' / 'frames'
            if not gt_clip_dir.exists():
                gt_clip_dir = gt_dir / clip_name / 'ground_truth'
            
            if not gt_clip_dir.exists():
                continue
            
            # Generate comparison ID
            comparison_id = f"{random.randint(1000, 9999)}_{clip_name[:20]}"
            
            output_video = output_dir / f"comparison_{comparison_id}.mp4"
            
            print(f"  Creating: {output_video.name}")
            
            # Try to create comparison
            order = create_comparison_from_frames(
                model_frames_dir=model_clip_dir,
                gt_frames_dir=gt_clip_dir,
                output_path=output_video,
                randomize_position=True,
            )
            
            if order:
                comparisons.append(ComparisonEntry(
                    comparison_id=comparison_id,
                    model_name=model_name,
                    clip_name=clip_name,
                    order=order,
                    video_path=str(output_video),
                ))
    
    # Shuffle comparisons
    random.shuffle(comparisons)
    
    # Generate forms
    print(f"\nGenerated {len(comparisons)} comparisons")
    
    form_path = output_dir / 'participant_form.md'
    generate_participant_form(comparisons, form_path)
    print("  ✓ Saved participant_form.md")
    
    key_path = output_dir / 'ANSWER_KEY_SECRET.json'
    generate_answer_key(comparisons, key_path)
    print("  ✓ Saved ANSWER_KEY_SECRET.json (DO NOT SHARE)")
    
    # Summary
    print("\n" + "=" * 60)
    print(" BLIND STUDY READY")
    print("=" * 60)
    print(f"Comparisons: {len(comparisons)}")
    print(f"Output directory: {output_dir}")
    print()
    print("Files generated:")
    print(f"  - {len(comparisons)} comparison videos")
    print("  - participant_form.md (give to participants)")
    print("  - ANSWER_KEY_SECRET.json (keep secret until after study)")
    print()
    print("Next steps:")
    print("  1. Share comparison videos with participants")
    print("  2. Have them fill out participant_form.md")
    print("  3. After all responses, use ANSWER_KEY_SECRET.json to analyze")


if __name__ == '__main__':
    main()
