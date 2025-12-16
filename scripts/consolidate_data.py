#!/usr/bin/env python3
"""
Consolidate all benchmark data into a single master file.
Gathers data from multiple sources and creates a unified dataset for reporting.
"""

import json
import os
from datetime import datetime
from pathlib import Path


def load_json_safe(path):
    """Load JSON file, return None if missing or invalid."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None


def consolidate_all_data(project_root):
    """Consolidate all benchmark data into one structure."""

    outputs_dir = project_root / 'outputs'

    consolidated = {
        'consolidated_at': datetime.now().isoformat(),
        'sources': [],
        'benchmark_runs': [],
        'methods_summary': {},
        'all_clips_metadata': []
    }

    # Source 1: outputs/benchmark/clips_metadata.json (latest 10s clips)
    path1 = outputs_dir / 'benchmark' / 'clips_metadata.json'
    data1 = load_json_safe(path1)
    if data1:
        consolidated['sources'].append(str(path1))
        run = {
            'source_file': str(path1),
            'type': '10s_clips_with_audio',
            'resolution': data1.get('resolution'),
            'input_resolution': data1.get('input_resolution'),
            'fps': data1.get('fps'),
            'duration_seconds': data1.get('duration_seconds'),
            'start_time': data1.get('start_time'),
            'clips': data1.get('clips', [])
        }
        consolidated['benchmark_runs'].append(run)
        consolidated['all_clips_metadata'].extend(data1.get('clips', []))
        print(f"  Loaded {len(data1.get('clips', []))} clips from {path1}")

    # Source 2: outputs/benchmark/benchmark_results.json
    path2 = outputs_dir / 'benchmark' / 'benchmark_results.json'
    data2 = load_json_safe(path2)
    if data2:
        consolidated['sources'].append(str(path2))
        run = {
            'source_file': str(path2),
            'type': 'benchmark_results',
            'methods': data2.get('methods', []),
            'by_quality': data2.get('by_quality', []),
            'by_speed': data2.get('by_speed', [])
        }
        consolidated['benchmark_runs'].append(run)
        print(f"  Loaded benchmark results with {len(data2.get('methods', []))} methods from {path2}")

    # Source 3: outputs/benchmarks/benchmark_results.json (detailed per-clip)
    path3 = outputs_dir / 'benchmarks' / 'benchmark_results.json'
    data3 = load_json_safe(path3)
    if data3:
        consolidated['sources'].append(str(path3))
        consolidated['detailed_per_clip_results'] = data3
        print(f"  Loaded detailed results for {len(data3)} clips from {path3}")

    # Source 4: outputs/blind_study_videos/clips_metadata.json
    path4 = outputs_dir / 'blind_study_videos' / 'clips_metadata.json'
    data4 = load_json_safe(path4)
    if data4:
        consolidated['sources'].append(str(path4))
        run = {
            'source_file': str(path4),
            'type': 'blind_study_videos',
            'models': data4.get('models', []),
            'clips': data4.get('clips', [])
        }
        consolidated['benchmark_runs'].append(run)
        print(f"  Loaded {len(data4.get('clips', []))} blind study clips from {path4}")

    # Source 5: outputs/blind_study_videos/sota_metadata.json
    path5 = outputs_dir / 'blind_study_videos' / 'sota_metadata.json'
    data5 = load_json_safe(path5)
    if data5:
        consolidated['sources'].append(str(path5))
        consolidated['sota_metadata'] = data5
        print(f"  Loaded SOTA metadata from {path5}")

    # Build methods summary
    methods_seen = {}
    for run in consolidated['benchmark_runs']:
        for clip in run.get('clips', []):
            method = clip.get('model') or clip.get('method')
            if method:
                if method not in methods_seen:
                    methods_seen[method] = {
                        'name': method,
                        'measurements': []
                    }
                methods_seen[method]['measurements'].append({
                    'source': run.get('source_file'),
                    'psnr_db': clip.get('psnr_db'),
                    'ssim': clip.get('ssim'),
                    'processing_time': clip.get('processing_time'),
                    'bitrate_mbps': clip.get('bitrate_mbps'),
                    'vfi_method': clip.get('vfi_method'),
                    'file_size_mb': clip.get('file_size_mb')
                })

    # Calculate averages for each method
    for method, data in methods_seen.items():
        measurements = data['measurements']
        psnr_vals = [m['psnr_db'] for m in measurements if m.get('psnr_db')]
        ssim_vals = [m['ssim'] for m in measurements if m.get('ssim')]
        time_vals = [m['processing_time'] for m in measurements if m.get('processing_time')]

        data['avg_psnr'] = sum(psnr_vals) / len(psnr_vals) if psnr_vals else None
        data['avg_ssim'] = sum(ssim_vals) / len(ssim_vals) if ssim_vals else None
        data['avg_time'] = sum(time_vals) / len(time_vals) if time_vals else None
        data['num_measurements'] = len(measurements)

    consolidated['methods_summary'] = methods_seen

    # Summary stats
    consolidated['summary'] = {
        'total_sources': len(consolidated['sources']),
        'total_benchmark_runs': len(consolidated['benchmark_runs']),
        'total_methods': len(methods_seen),
        'methods_list': list(methods_seen.keys())
    }

    return consolidated


def main():
    project_root = Path(__file__).parent.parent
    outputs_dir = project_root / 'outputs'

    print("=" * 60)
    print("DATA CONSOLIDATION")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Scanning for data files...\n")

    consolidated = consolidate_all_data(project_root)

    # Save consolidated data
    output_path = outputs_dir / 'all_benchmark_data.json'
    with open(output_path, 'w') as f:
        json.dump(consolidated, f, indent=2)

    print(f"\n{'=' * 60}")
    print("CONSOLIDATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Sources found: {consolidated['summary']['total_sources']}")
    print(f"Benchmark runs: {consolidated['summary']['total_benchmark_runs']}")
    print(f"Methods tracked: {consolidated['summary']['total_methods']}")
    print(f"Methods: {', '.join(consolidated['summary']['methods_list'])}")
    print(f"\nOutput saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()
