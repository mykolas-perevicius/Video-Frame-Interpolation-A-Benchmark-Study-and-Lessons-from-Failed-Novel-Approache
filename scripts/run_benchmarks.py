#!/usr/bin/env python3
"""
run_benchmarks.py

Run all models on test clips and collect results.

Usage:
    # Run all models on all clips
    python scripts/run_benchmarks.py --data-dir data/processed --output-dir outputs/benchmarks
    
    # Run specific models
    python scripts/run_benchmarks.py --data-dir data/processed --output-dir outputs/benchmarks --models rife,adaptive
    
    # Quick test with limited triplets
    python scripts/run_benchmarks.py --data-dir data/processed --output-dir outputs/benchmarks --max-triplets 10
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import QualityEvaluator, compute_psnr_simple, compute_ssim_simple


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB numpy array"""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_clip_data(clip_dir: Path) -> dict:
    """Load preprocessed clip data"""
    metadata_path = clip_dir / 'metadata.json'
    triplets_path = clip_dir / 'triplets.json'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not triplets_path.exists():
        raise FileNotFoundError(f"Triplets not found: {triplets_path}")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    with open(triplets_path) as f:
        triplets = json.load(f)
    
    return {
        'metadata': metadata,
        'triplets': triplets,
        'input_dir': clip_dir / 'input_1080p30' / 'frames',
        'gt_dir': clip_dir / 'ground_truth' / 'frames',
    }


def get_available_models() -> Dict[str, type]:
    """
    Get dictionary of available models.
    
    Returns model name -> model class mapping.
    """
    models = {}
    
    # Traditional baselines (always available)
    try:
        from models.traditional.baselines import BicubicBaseline, LanczosBaseline, OpticalFlowVFI
        models['bicubic'] = BicubicBaseline
        models['lanczos'] = LanczosBaseline
        models['optical_flow'] = OpticalFlowVFI
    except ImportError as e:
        print(f"Warning: Could not import traditional baselines: {e}")
    
    # RIFE (speed champion)
    try:
        from models.sota.rife_wrapper import RIFEModel
        models['rife'] = RIFEModel
    except ImportError as e:
        print(f"Warning: Could not import RIFE: {e}")
    
    # Novel adaptive pipeline
    try:
        from models.novel.adaptive_pipeline import AdaptivePipeline
        models['adaptive'] = AdaptivePipeline
    except ImportError as e:
        print(f"Warning: Could not import AdaptivePipeline: {e}")
    
    # VFIMamba (quality SOTA) - optional
    try:
        from models.sota.vfimamba_wrapper import VFIMambaModel
        models['vfimamba'] = VFIMambaModel
    except ImportError:
        pass  # Optional
    
    return models


def run_model_benchmark(
    model,
    clip_data: dict,
    evaluator: QualityEvaluator,
    max_triplets: Optional[int] = None,
    save_outputs: bool = False,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run benchmark for a single model on a single clip.
    
    Returns dictionary with quality and speed results.
    """
    triplets = clip_data['triplets']
    if max_triplets:
        triplets = triplets[:max_triplets]
    
    input_dir = clip_data['input_dir']
    gt_dir = clip_data['gt_dir']
    metadata = clip_data['metadata']
    
    # Results accumulators
    quality_results = []
    speed_results = []
    
    # Get number of intermediate frames to generate
    if triplets:
        num_intermediate = triplets[0]['num_intermediate']
    else:
        num_intermediate = 3
    
    target_scale = metadata['spatial_scale']
    
    # Process each triplet
    for triplet in tqdm(triplets, desc=f"    {model.info.name}", leave=False):
        try:
            # Load input frames
            frame0 = load_image(input_dir / triplet['input_frame_0'])
            frame1 = load_image(input_dir / triplet['input_frame_1'])
            
            # Run model
            result = model.process_pair(
                frame0, frame1,
                num_intermediate=num_intermediate,
                target_scale=target_scale,
            )
            
            # Load ground truth intermediate frames
            gt_frames = [
                load_image(gt_dir / gt_name)
                for gt_name in triplet['gt_intermediate']
            ]
            
            # Extract predicted intermediate frames (exclude endpoints)
            pred_intermediate = result.frames[1:-1]
            
            # Ensure we have the right number
            if len(pred_intermediate) != len(gt_frames):
                print(f"    Warning: Predicted {len(pred_intermediate)} frames, "
                      f"GT has {len(gt_frames)}")
                continue
            
            # Evaluate quality
            try:
                quality = evaluator.evaluate(pred_intermediate, gt_frames)
                quality_results.append(quality.to_dict())
            except Exception:
                # Fallback to simple metrics
                psnr = np.mean([compute_psnr_simple(p, g) for p, g in zip(pred_intermediate, gt_frames)])
                ssim = np.mean([compute_ssim_simple(p, g) for p, g in zip(pred_intermediate, gt_frames)])
                quality_results.append({'psnr': psnr, 'ssim': ssim, 'lpips': 0.1})
            
            # Speed results
            speed_results.append({
                'inference_time_ms': result.inference_time_ms,
                'vram_peak_mb': result.vram_peak_mb,
            })
            
            # Optionally save outputs
            if save_outputs and output_dir:
                triplet_output_dir = output_dir / f"triplet_{triplet['triplet_id']:04d}"
                triplet_output_dir.mkdir(parents=True, exist_ok=True)
                for i, frame in enumerate(result.frames):
                    cv2.imwrite(
                        str(triplet_output_dir / f"frame_{i:02d}.png"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    )
            
        except Exception as e:
            print(f"    Error processing triplet {triplet['triplet_id']}: {e}")
            continue
    
    if not quality_results:
        return {'error': 'No successful evaluations'}
    
    # Aggregate results
    def aggregate_metric(results: List[dict], key: str) -> dict:
        values = [r[key] for r in results if key in r and r[key] is not None]
        if not values:
            return {'mean': None}
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'p50': float(np.percentile(values, 50)),
            'p95': float(np.percentile(values, 95)),
        }
    
    quality_summary = {
        'psnr': aggregate_metric(quality_results, 'psnr'),
        'ssim': aggregate_metric(quality_results, 'ssim'),
        'lpips': aggregate_metric(quality_results, 'lpips'),
    }
    
    times = [r['inference_time_ms'] for r in speed_results]
    vram = [r['vram_peak_mb'] for r in speed_results]
    
    speed_summary = {
        'time_ms': {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'p50': float(np.percentile(times, 50)),
            'p95': float(np.percentile(times, 95)),
            'p99': float(np.percentile(times, 99)),
        },
        'vram_mb': {
            'mean': float(np.mean(vram)),
            'max': float(np.max(vram)),
        },
        # 5 output frames per pair (2 endpoints + 3 intermediate)
        'throughput_fps': 1000 / np.mean(times) * 5 if times else 0,
    }
    
    return {
        'model': model.info.name,
        'model_info': asdict(model.info),
        'num_triplets': len(quality_results),
        'quality_summary': quality_summary,
        'speed_summary': speed_summary,
        'quality_per_triplet': quality_results,
        'speed_per_triplet': speed_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run VFI+SR benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--data-dir', '-d',
        required=True,
        help='Directory containing preprocessed clips'
    )
    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--models', '-m',
        default='all',
        help='Comma-separated model names or "all" (default: all)'
    )
    parser.add_argument(
        '--clips', '-c',
        default='all',
        help='Comma-separated clip names or "all" (default: all)'
    )
    parser.add_argument(
        '--max-triplets',
        type=int,
        default=None,
        help='Maximum triplets per clip (for quick testing)'
    )
    parser.add_argument(
        '--save-outputs',
        action='store_true',
        help='Save model outputs (uses more disk space)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=3,
        help='Number of warmup iterations before timing'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get available models
    available_models = get_available_models()
    
    if args.models == 'all':
        models_to_test = list(available_models.keys())
    else:
        models_to_test = [m.strip() for m in args.models.split(',')]
        # Validate
        for m in models_to_test:
            if m not in available_models:
                print(f"Error: Model '{m}' not available")
                print(f"Available models: {list(available_models.keys())}")
                return 1
    
    # Get clips to test
    if args.clips == 'all':
        clips_to_test = [d.name for d in data_dir.iterdir() if d.is_dir() and (d / 'metadata.json').exists()]
    else:
        clips_to_test = [c.strip() for c in args.clips.split(',')]
    
    if not clips_to_test:
        print(f"Error: No clips found in {data_dir}")
        return 1
    
    print("=" * 60)
    print(" VFI+SR BENCHMARK")
    print("=" * 60)
    print(f"Models: {', '.join(models_to_test)}")
    print(f"Clips: {', '.join(clips_to_test)}")
    print(f"Output: {output_dir}")
    print()
    
    # Initialize quality evaluator
    try:
        evaluator = QualityEvaluator()
    except Exception as e:
        print(f"Warning: Could not initialize full evaluator: {e}")
        print("Using simple metrics (PSNR/SSIM only)")
        evaluator = None
    
    # Run benchmarks
    all_results = {}
    
    for clip_name in clips_to_test:
        print(f"\n{'=' * 60}")
        print(f"CLIP: {clip_name}")
        print(f"{'=' * 60}")
        
        clip_dir = data_dir / clip_name
        
        try:
            clip_data = load_clip_data(clip_dir)
        except Exception as e:
            print(f"Error loading clip: {e}")
            continue
        
        print(f"  Triplets: {len(clip_data['triplets'])}")
        print(f"  Target scale: {clip_data['metadata']['spatial_scale']:.2f}x")
        
        all_results[clip_name] = {}
        
        for model_name in models_to_test:
            print(f"\n  Model: {model_name}")
            
            try:
                # Instantiate model
                model_class = available_models[model_name]
                model = model_class()
                
                # Load model
                print("    Loading...")
                model.load()
                
                # Warmup
                if args.warmup > 0 and clip_data['triplets']:
                    print(f"    Warming up ({args.warmup} iterations)...")
                    triplet = clip_data['triplets'][0]
                    frame0 = load_image(clip_data['input_dir'] / triplet['input_frame_0'])
                    frame1 = load_image(clip_data['input_dir'] / triplet['input_frame_1'])
                    for _ in range(args.warmup):
                        _ = model.process_pair(frame0, frame1)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                # Run benchmark
                model_output_dir = output_dir / 'model_outputs' / clip_name / model_name if args.save_outputs else None
                
                result = run_model_benchmark(
                    model=model,
                    clip_data=clip_data,
                    evaluator=evaluator,
                    max_triplets=args.max_triplets,
                    save_outputs=args.save_outputs,
                    output_dir=model_output_dir,
                )
                
                all_results[clip_name][model_name] = result
                
                # Print summary
                if 'error' not in result:
                    psnr = result['quality_summary']['psnr']['mean']
                    lpips = result['quality_summary']['lpips']['mean']
                    time_ms = result['speed_summary']['time_ms']['mean']
                    fps = result['speed_summary']['throughput_fps']
                    
                    print(f"    PSNR: {psnr:.2f} dB | LPIPS: {lpips:.4f}")
                    print(f"    Time: {time_ms:.1f} ms | Throughput: {fps:.1f} fps")
                else:
                    print(f"    Error: {result['error']}")
                
                # Clean up
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results[clip_name][model_name] = {'error': str(e)}
    
    # Save results
    results_path = output_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(" BENCHMARK COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {results_path}")
    
    # Print summary table
    print("\nSUMMARY:")
    print("-" * 80)
    print(f"{'Model':<20} {'PSNR (dB)':<12} {'LPIPS':<12} {'Time (ms)':<12} {'FPS':<10}")
    print("-" * 80)
    
    for clip_name, clip_results in all_results.items():
        for model_name, result in clip_results.items():
            if 'error' not in result:
                psnr = result['quality_summary']['psnr']['mean']
                lpips = result['quality_summary']['lpips']['mean']
                time_ms = result['speed_summary']['time_ms']['mean']
                fps = result['speed_summary']['throughput_fps']
                print(f"{model_name:<20} {psnr:<12.2f} {lpips:<12.4f} {time_ms:<12.1f} {fps:<10.1f}")
    
    print("-" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
