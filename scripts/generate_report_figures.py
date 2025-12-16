#!/usr/bin/env python3
"""Generate publication-quality figures for the VFI+SR report."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

# Color scheme by category
COLORS = {
    'reference': '#2ecc71',      # Green
    'baseline': '#e74c3c',       # Red
    'traditional': '#3498db',    # Blue
    'neural': '#9b59b6',         # Purple
    'adaptive': '#f39c12',       # Orange
    'novel': '#1abc9c',          # Teal
}

CATEGORIES = {
    'control': 'reference',
    'degraded': 'baseline',
    'lanczos_blend': 'traditional',
    'lanczos_blend_edge': 'traditional',
    'lanczos_blend_sharp': 'traditional',
    'bicubic_blend': 'traditional',
    'bicubic_blend_edge': 'traditional',
    'optical_flow_basic': 'traditional',
    'optical_flow_edge': 'traditional',
    'optical_flow_sharp': 'traditional',
    'rife_default': 'neural',
    'rife_fast': 'neural',
    'adaptive_conservative': 'adaptive',
    'adaptive_default': 'adaptive',
    'adaptive_aggressive': 'adaptive',
    'mcar_default': 'novel',
    'mcar_aggressive': 'novel',
    'uafi_default': 'novel',
    'ughi_default': 'novel',
}

def load_results():
    results_file = Path(__file__).parent.parent / 'outputs' / 'experiment_results.json'
    with open(results_file) as f:
        return json.load(f)['experiments']

def fig1_vfi_quality_comparison(experiments, output_dir):
    """Bar chart comparing VFI PSNR across all methods."""
    # Filter out control (VFI=0) and sort by VFI PSNR
    data = [(e['name'], e.get('vfi_psnr_db', 0), CATEGORIES.get(e['name'], 'other'))
            for e in experiments if e.get('vfi_psnr_db', 0) > 0]
    data.sort(key=lambda x: x[1], reverse=True)

    names = [d[0].replace('_', '\n') for d in data]
    values = [d[1] for d in data]
    colors = [COLORS.get(d[2], '#95a5a6') for d in data]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # Add baseline reference line
    baseline = next((e.get('vfi_psnr_db', 0) for e in experiments if e['name'] == 'degraded'), 22.28)
    ax.axhline(y=baseline, color='red', linestyle='--', linewidth=1.5, label=f'Baseline ({baseline:.2f}dB)')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('VFI PSNR (dB)')
    ax.set_title('Frame Interpolation Quality Comparison\n(Higher is Better)')
    ax.set_ylim(20, 27)

    # Legend
    legend_patches = [mpatches.Patch(color=COLORS[cat], label=cat.title())
                      for cat in ['baseline', 'traditional', 'neural', 'adaptive', 'novel']]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_vfi_quality_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_vfi_quality_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig1_vfi_quality_comparison.png/pdf")

def fig2_quality_vs_speed(experiments, output_dir):
    """Scatter plot of quality vs processing time."""
    data = [(e['name'], e.get('vfi_psnr_db', 0), e.get('time_s', 0), CATEGORIES.get(e['name'], 'other'))
            for e in experiments if e.get('vfi_psnr_db', 0) > 0]

    fig, ax = plt.subplots(figsize=(10, 7))

    for name, vfi, time, cat in data:
        color = COLORS.get(cat, '#95a5a6')
        ax.scatter(time, vfi, c=color, s=100, edgecolors='black', linewidth=0.5, alpha=0.8)
        # Label key methods
        if name in ['rife_default', 'lanczos_blend', 'degraded', 'adaptive_aggressive', 'ughi_default']:
            ax.annotate(name.replace('_', ' '), (time, vfi),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xlabel('Processing Time (seconds)')
    ax.set_ylabel('VFI PSNR (dB)')
    ax.set_title('Quality vs Speed Tradeoff\n(Upper-left is better)')

    # Add quadrant labels
    ax.axhline(y=24, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=60, color='gray', linestyle=':', alpha=0.5)

    # Legend
    legend_patches = [mpatches.Patch(color=COLORS[cat], label=cat.title())
                      for cat in ['baseline', 'traditional', 'neural', 'adaptive', 'novel']]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_quality_vs_speed.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_quality_vs_speed.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig2_quality_vs_speed.png/pdf")

def fig3_category_comparison(experiments, output_dir):
    """Grouped bar chart comparing categories."""
    # Group by category
    categories = {}
    for e in experiments:
        cat = CATEGORIES.get(e['name'], 'other')
        if cat == 'reference':
            continue
        if cat not in categories:
            categories[cat] = []
        if e.get('vfi_psnr_db', 0) > 0:
            categories[cat].append(e.get('vfi_psnr_db', 0))

    # Calculate stats
    cat_names = ['baseline', 'traditional', 'neural', 'adaptive', 'novel']
    means = [np.mean(categories.get(cat, [0])) for cat in cat_names]
    stds = [np.std(categories.get(cat, [0])) if len(categories.get(cat, [])) > 1 else 0 for cat in cat_names]
    colors = [COLORS[cat] for cat in cat_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cat_names, means, yerr=stds, color=colors, edgecolor='black',
                  linewidth=0.5, capsize=5, error_kw={'linewidth': 1.5})

    # Add value labels
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}dB', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Average VFI PSNR (dB)')
    ax.set_title('VFI Quality by Method Category')
    ax.set_ylim(20, 27)

    # Capitalize labels
    ax.set_xticklabels([cat.title() for cat in cat_names])

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_category_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_category_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig3_category_comparison.png/pdf")

def fig4_rife_usage_impact(experiments, output_dir):
    """Show correlation between RIFE usage % and quality for adaptive methods."""
    adaptive = [(e['name'], e.get('vfi_psnr_db', 0), e.get('rife_frames_pct', 0))
                for e in experiments if 'adaptive' in e['name']]

    fig, ax = plt.subplots(figsize=(8, 5))

    names = [a[0].replace('adaptive_', '') for a in adaptive]
    vfi_psnr = [a[1] for a in adaptive]
    rife_pct = [a[2] for a in adaptive]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, vfi_psnr, width, label='VFI PSNR (dB)', color=COLORS['adaptive'])
    bars2 = ax.bar(x + width/2, [r/4 + 20 for r in rife_pct], width, label='RIFE Usage (scaled)',
                   color=COLORS['neural'], alpha=0.7)

    ax.set_ylabel('VFI PSNR (dB)')
    ax.set_title('Adaptive Methods: RIFE Usage vs Quality')
    ax.set_xticks(x)
    ax.set_xticklabels([n.title() for n in names])
    ax.legend()
    ax.set_ylim(20, 27)

    # Add RIFE % labels
    for i, (bar, pct) in enumerate(zip(bars2, rife_pct)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{pct:.0f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_rife_usage_impact.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_rife_usage_impact.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig4_rife_usage_impact.png/pdf")

def table1_full_results(experiments, output_dir):
    """Generate LaTeX table of all results."""
    # Sort by VFI PSNR
    data = sorted([e for e in experiments if e.get('vfi_psnr_db', 0) > 0],
                  key=lambda x: x.get('vfi_psnr_db', 0), reverse=True)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Complete VFI+SR Benchmark Results}
\label{tab:results}
\begin{tabular}{llcccc}
\toprule
\textbf{Rank} & \textbf{Method} & \textbf{VFI PSNR} & \textbf{KF PSNR} & \textbf{Time (s)} & \textbf{RIFE \%} \\
\midrule
"""

    for i, e in enumerate(data, 1):
        name = e['name'].replace('_', r'\_')
        vfi = e.get('vfi_psnr_db', 0)
        kf = e.get('keyframe_psnr_db', 0)
        time = e.get('time_s', 0)
        rife = e.get('rife_frames_pct', 0)
        latex += f"{i} & {name} & {vfi:.2f} dB & {kf:.2f} dB & {time:.1f} & {rife:.0f}\\% \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_dir / 'table1_results.tex', 'w') as f:
        f.write(latex)
    print(f"  Saved: table1_results.tex")

def main():
    print("\n" + "="*60)
    print("GENERATING REPORT FIGURES")
    print("="*60)

    output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    output_dir.mkdir(exist_ok=True)

    experiments = load_results()
    print(f"Loaded {len(experiments)} experiments\n")

    print("Generating figures...")
    fig1_vfi_quality_comparison(experiments, output_dir)
    fig2_quality_vs_speed(experiments, output_dir)
    fig3_category_comparison(experiments, output_dir)
    fig4_rife_usage_impact(experiments, output_dir)
    table1_full_results(experiments, output_dir)

    print("\n" + "="*60)
    print(f"All figures saved to: {output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
