#!/usr/bin/env python3
"""
generate_visualizations.py

Create publication-quality figures from benchmark results.

Usage:
    python scripts/generate_visualizations.py --results outputs/benchmarks/benchmark_results.json --output outputs/visualizations
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def load_results(results_path: Path) -> dict:
    """Load benchmark results from JSON"""
    with open(results_path) as f:
        return json.load(f)


def aggregate_model_stats(results: dict) -> dict:
    """Aggregate results across all clips for each model"""
    model_stats = {}
    
    for clip_name, clip_results in results.items():
        for model_name, model_results in clip_results.items():
            if 'error' in model_results:
                continue
            
            if model_name not in model_stats:
                model_stats[model_name] = {
                    'time_ms': [],
                    'lpips': [],
                    'psnr': [],
                    'ssim': [],
                    'vram_mb': [],
                }
            
            # Extract metrics
            if 'speed_summary' in model_results:
                model_stats[model_name]['time_ms'].append(
                    model_results['speed_summary']['time_ms']['mean']
                )
                model_stats[model_name]['vram_mb'].append(
                    model_results['speed_summary']['vram_mb']['max']
                )
            
            if 'quality_summary' in model_results:
                q = model_results['quality_summary']
                if 'lpips' in q and q['lpips']['mean'] is not None:
                    model_stats[model_name]['lpips'].append(q['lpips']['mean'])
                if 'psnr' in q and q['psnr']['mean'] is not None:
                    model_stats[model_name]['psnr'].append(q['psnr']['mean'])
                if 'ssim' in q and q['ssim']['mean'] is not None:
                    model_stats[model_name]['ssim'].append(q['ssim']['mean'])
    
    return model_stats


def get_model_color(model_name: str) -> str:
    """Get consistent color for model"""
    colors = {
        # Traditional (grays)
        'bicubic': '#999999',
        'lanczos': '#666666',
        'optical_flow': '#444444',
        
        # SOTA (blues/greens)
        'rife': '#2ecc71',
        'rife_span': '#27ae60',
        'vfimamba': '#3498db',
        'vfimamba_span': '#2980b9',
        'safa': '#e74c3c',
        'span': '#1abc9c',
        
        # Novel (purples) - HIGHLIGHT THESE
        'adaptive': '#9b59b6',
        'adaptivepipeline': '#9b59b6',
        'cascade_720p': '#8e44ad',
        'cascade_1080p': '#7d3c98',
    }
    
    # Check for partial matches
    model_lower = model_name.lower()
    for key, color in colors.items():
        if key in model_lower:
            return color
    
    return '#34495e'  # Default gray-blue


def get_model_marker(model_name: str) -> str:
    """Get marker style based on model type"""
    model_lower = model_name.lower()
    
    if any(t in model_lower for t in ['bicubic', 'lanczos', 'optical']):
        return 's'  # Square for traditional
    elif any(t in model_lower for t in ['adaptive', 'cascade', 'novel']):
        return '*'  # Star for novel (ours)
    else:
        return 'o'  # Circle for SOTA


def plot_pareto_frontier(model_stats: dict, output_dir: Path):
    """
    Create Pareto frontier plot: Speed vs Quality
    This is THE key figure for the paper.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping Pareto plot - matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each model
    points = []
    for model_name, stats in model_stats.items():
        if not stats['time_ms'] or not stats['lpips']:
            continue
        
        avg_time = np.mean(stats['time_ms'])
        avg_lpips = np.mean(stats['lpips'])
        
        color = get_model_color(model_name)
        marker = get_model_marker(model_name)
        
        # Size based on type (novel = larger)
        size = 400 if '*' in marker else 150
        
        ax.scatter(
            avg_time, avg_lpips,
            c=color,
            s=size,
            marker=marker,
            label=model_name,
            alpha=0.85,
            edgecolors='white',
            linewidths=2,
            zorder=10 if '*' in marker else 5,
        )
        
        points.append((avg_time, avg_lpips, model_name))
    
    # Draw Pareto frontier
    if points:
        points.sort(key=lambda x: x[0])  # Sort by time
        
        pareto_points = []
        min_lpips = float('inf')
        for time, lpips, name in points:
            if lpips < min_lpips:
                pareto_points.append((time, lpips))
                min_lpips = lpips
        
        if len(pareto_points) > 1:
            pareto_x = [p[0] for p in pareto_points]
            pareto_y = [p[1] for p in pareto_points]
            ax.plot(pareto_x, pareto_y, 'k--', alpha=0.4, linewidth=2, 
                    label='Pareto Frontier', zorder=1)
    
    # Real-time threshold line
    # For 30fps -> 120fps with 3 intermediate frames: need to process in < 33.33ms
    ax.axvline(x=33.33, color='red', linestyle=':', alpha=0.7, linewidth=2,
               label='Real-time threshold (33.3ms)')
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Processing Time per Frame Pair (ms) - Log Scale', fontsize=13)
    ax.set_ylabel('LPIPS ↓ (Lower is Better)', fontsize=13)
    ax.set_title('Speed vs Quality Pareto Analysis\nGaming VFI+SR Benchmark', fontsize=15, fontweight='bold')
    
    # Legend
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    ax.grid(True, alpha=0.3, which='both')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontier.png', dpi=200, bbox_inches='tight')
    plt.savefig(output_dir / 'pareto_frontier.pdf', bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved pareto_frontier.png/pdf")


def plot_speed_comparison(model_stats: dict, output_dir: Path):
    """Bar chart comparing processing speeds"""
    if not HAS_MATPLOTLIB:
        return
    
    # Prepare data
    models = []
    means = []
    stds = []
    colors = []
    
    for model_name, stats in model_stats.items():
        if not stats['time_ms']:
            continue
        models.append(model_name)
        means.append(np.mean(stats['time_ms']))
        stds.append(np.std(stats['time_ms']))
        colors.append(get_model_color(model_name))
    
    # Sort by speed (fastest first)
    sorted_indices = np.argsort(means)
    models = [models[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 0.5)))
    
    y_pos = np.arange(len(models))
    ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.85,
            edgecolor='white', linewidth=1)
    
    # Real-time line
    ax.axvline(x=33.33, color='red', linestyle='--', linewidth=2,
               label='Real-time (30→120fps)')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Processing Time (ms) per Frame Pair', fontsize=12)
    ax.set_title('Speed Comparison Across Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speed_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved speed_comparison.png")


def plot_quality_comparison(model_stats: dict, output_dir: Path):
    """Bar chart comparing quality metrics"""
    if not HAS_MATPLOTLIB:
        return
    
    metrics = ['psnr', 'lpips']
    
    for metric in metrics:
        models = []
        values = []
        colors = []
        
        for model_name, stats in model_stats.items():
            if not stats.get(metric):
                continue
            models.append(model_name)
            values.append(np.mean(stats[metric]))
            colors.append(get_model_color(model_name))
        
        if not models:
            continue
        
        # Sort appropriately (PSNR: higher is better, LPIPS: lower is better)
        if metric == 'psnr':
            sorted_indices = np.argsort(values)[::-1]  # Descending
        else:
            sorted_indices = np.argsort(values)  # Ascending
        
        models = [models[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 0.5)))
        
        y_pos = np.arange(len(models))
        ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor='white')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        
        if metric == 'psnr':
            ax.set_xlabel('PSNR (dB) ↑ Higher is Better', fontsize=12)
        else:
            ax.set_xlabel('LPIPS ↓ Lower is Better', fontsize=12)
        
        ax.set_title(f'{metric.upper()} Quality Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved {metric}_comparison.png")


def generate_summary_table(model_stats: dict, output_dir: Path):
    """Generate markdown summary table"""
    
    lines = [
        "# Benchmark Summary\n",
        "",
        "| Model | PSNR (dB) ↑ | LPIPS ↓ | Time (ms) ↓ | VRAM (MB) | Throughput (fps) |",
        "|-------|-------------|---------|-------------|-----------|------------------|",
    ]
    
    # Prepare data for sorting
    model_data = []
    for model_name, stats in model_stats.items():
        if not stats['time_ms']:
            continue
        
        psnr = np.mean(stats['psnr']) if stats['psnr'] else 0
        lpips = np.mean(stats['lpips']) if stats['lpips'] else 1
        time_ms = np.mean(stats['time_ms'])
        vram = np.mean(stats['vram_mb']) if stats['vram_mb'] else 0
        fps = 1000 / time_ms * 5 if time_ms > 0 else 0  # 5 output frames per pair
        
        model_data.append({
            'name': model_name,
            'psnr': psnr,
            'lpips': lpips,
            'time_ms': time_ms,
            'vram': vram,
            'fps': fps,
        })
    
    # Sort by LPIPS (best quality first)
    model_data.sort(key=lambda x: x['lpips'])
    
    for m in model_data:
        # Highlight novel methods
        name = m['name']
        if 'adaptive' in name.lower() or 'cascade' in name.lower():
            name = f"**{name}**"
        
        lines.append(
            f"| {name} | {m['psnr']:.2f} | {m['lpips']:.4f} | "
            f"{m['time_ms']:.1f} | {m['vram']:.0f} | {m['fps']:.1f} |"
        )
    
    # Add notes
    lines.extend([
        "",
        "## Notes",
        "- **Bold** = Novel methods (ours)",
        "- PSNR: Peak Signal-to-Noise Ratio (higher is better)",
        "- LPIPS: Learned Perceptual Image Patch Similarity (lower is better)",
        "- Throughput: Output frames per second (5 frames per input pair)",
        "- Real-time threshold: 33.3ms per pair for 30fps→120fps",
    ])
    
    with open(output_dir / 'summary_table.md', 'w') as f:
        f.write('\n'.join(lines))
    
    print("  ✓ Saved summary_table.md")


def generate_latex_table(model_stats: dict, output_dir: Path):
    """Generate LaTeX table for academic report"""
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Benchmark Results for Gaming VFI+SR}",
        r"\label{tab:benchmark}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & PSNR $\uparrow$ & LPIPS $\downarrow$ & Time (ms) & VRAM (MB) & FPS \\",
        r"\midrule",
    ]
    
    # Prepare and sort data
    model_data = []
    for model_name, stats in model_stats.items():
        if not stats['time_ms']:
            continue
        
        psnr = np.mean(stats['psnr']) if stats['psnr'] else 0
        lpips = np.mean(stats['lpips']) if stats['lpips'] else 1
        time_ms = np.mean(stats['time_ms'])
        vram = np.mean(stats['vram_mb']) if stats['vram_mb'] else 0
        fps = 1000 / time_ms * 5 if time_ms > 0 else 0
        
        model_data.append({
            'name': model_name,
            'psnr': psnr,
            'lpips': lpips,
            'time_ms': time_ms,
            'vram': vram,
            'fps': fps,
        })
    
    model_data.sort(key=lambda x: x['lpips'])
    
    for m in model_data:
        name = m['name'].replace('_', r'\_')
        # Bold for best, italic for novel
        if 'adaptive' in m['name'].lower():
            name = r"\textbf{" + name + "}"
        
        lines.append(
            f"{name} & {m['psnr']:.2f} & {m['lpips']:.4f} & "
            f"{m['time_ms']:.1f} & {m['vram']:.0f} & {m['fps']:.1f} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(output_dir / 'benchmark_table.tex', 'w') as f:
        f.write('\n'.join(lines))
    
    print("  ✓ Saved benchmark_table.tex")


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations from benchmark results'
    )
    parser.add_argument(
        '--results', '-r',
        required=True,
        help='Path to benchmark_results.json'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for figures'
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print(" GENERATING VISUALIZATIONS")
    print("=" * 50)
    
    # Load results
    print(f"\nLoading results from {results_path}...")
    results = load_results(results_path)
    
    # Aggregate stats
    print("Aggregating model statistics...")
    model_stats = aggregate_model_stats(results)
    print(f"  Found {len(model_stats)} models")
    
    # Generate visualizations
    print("\nGenerating figures...")
    plot_pareto_frontier(model_stats, output_dir)
    plot_speed_comparison(model_stats, output_dir)
    plot_quality_comparison(model_stats, output_dir)
    
    # Generate tables
    print("\nGenerating tables...")
    generate_summary_table(model_stats, output_dir)
    generate_latex_table(model_stats, output_dir)
    
    print("\n" + "=" * 50)
    print(f" DONE! Output saved to: {output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
