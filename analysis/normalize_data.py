#!/usr/bin/env python3
"""
Normalize experiment data across different video intervals.

Scientific approach:
1. Use 'degraded' baseline as reference point per interval
2. Report PSNR GAIN over baseline (not absolute PSNR)
3. Normalize timing relative to baseline
4. Flag cross-interval comparisons

This allows fair comparison even when methods were tested on different intervals.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = PROJECT_ROOT / 'outputs' / 'experiment_results.json'
OUTPUT_FILE = PROJECT_ROOT / 'outputs' / 'normalized_results.json'


def load_and_normalize():
    """Load raw data and create normalized dataset."""

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    df = pd.DataFrame(data['experiments'])

    # Extract interval info
    df['interval_start'] = df['interval'].apply(lambda x: x['start'])
    df['interval_end'] = df['interval'].apply(lambda x: x['end'])
    df['interval_id'] = df['interval_start'].astype(str) + '-' + df['interval_end'].astype(str)

    # Remove control (infinite PSNR)
    df = df[df['name'] != 'control'].copy()

    print("=" * 70)
    print("NORMALIZING DATA ACROSS INTERVALS")
    print("=" * 70)

    # Get baseline (degraded) for each interval
    baselines = {}
    for interval_id in df['interval_id'].unique():
        interval_df = df[df['interval_id'] == interval_id]
        degraded = interval_df[interval_df['name'] == 'degraded']
        if len(degraded) > 0:
            baselines[interval_id] = {
                'psnr': degraded['psnr_db'].values[0],
                'ssim': degraded['ssim'].values[0],
                'time': degraded['time_s'].values[0],
            }
            print(f"\nInterval {interval_id}s baseline (degraded):")
            print(f"  PSNR: {baselines[interval_id]['psnr']:.2f} dB")
            print(f"  SSIM: {baselines[interval_id]['ssim']:.4f}")

    # Normalize each experiment relative to its interval's baseline
    normalized_rows = []

    for _, row in df.iterrows():
        interval_id = row['interval_id']

        if interval_id not in baselines:
            continue

        baseline = baselines[interval_id]

        # Calculate normalized metrics
        psnr_gain = row['psnr_db'] - baseline['psnr']
        ssim_gain = row['ssim'] - baseline['ssim']
        time_ratio = row['time_s'] / baseline['time']

        # Efficiency: PSNR gain per unit time ratio
        efficiency = psnr_gain / time_ratio if time_ratio > 0 else 0

        normalized_rows.append({
            'name': row['name'],
            'interval_id': interval_id,
            'vfi_method': row.get('vfi_method', ''),

            # Absolute metrics
            'psnr_db': row['psnr_db'],
            'ssim': row['ssim'],
            'time_s': row['time_s'],

            # Normalized metrics (THE KEY DATA)
            'psnr_gain': psnr_gain,
            'ssim_gain': ssim_gain,
            'time_ratio': time_ratio,
            'efficiency': efficiency,

            # Baseline reference
            'baseline_psnr': baseline['psnr'],
            'baseline_ssim': baseline['ssim'],

            # Metadata
            'rife_frames_pct': row.get('rife_frames_pct', 0),
            'experiment_type': row.get('experiment_type', 'UNKNOWN'),
        })

    normalized_df = pd.DataFrame(normalized_rows)

    return normalized_df, baselines


def deduplicate_methods(df):
    """
    For methods tested on multiple intervals, keep the most representative.

    Strategy: Keep the run with median PSNR gain (most typical performance).
    """

    print("\n" + "=" * 70)
    print("DEDUPLICATING METHODS")
    print("=" * 70)

    deduped_rows = []

    for name in df['name'].unique():
        method_df = df[df['name'] == name]

        if len(method_df) == 1:
            deduped_rows.append(method_df.iloc[0].to_dict())
        else:
            # Multiple runs - pick median or note all
            print(f"\n  {name}: {len(method_df)} runs")
            for _, row in method_df.iterrows():
                print(f"    Interval {row['interval_id']}: gain={row['psnr_gain']:+.2f} dB")

            # Keep the one with median PSNR gain
            median_idx = method_df['psnr_gain'].argsort().values[len(method_df)//2]
            selected = method_df.iloc[median_idx]
            print(f"    -> Selected: {selected['interval_id']} (median)")

            deduped_rows.append(selected.to_dict())

    return pd.DataFrame(deduped_rows)


def categorize_methods(df):
    """Add method categories for analysis."""

    def get_category(row):
        name = row['name'].lower()
        vfi = str(row.get('vfi_method', '')).lower()

        if 'uafi' in name or 'ui_aware' in vfi:
            return 'UAFI (Novel)'
        elif 'mcar' in name:
            return 'MCAR (Novel)'
        elif 'ughi' in name:
            return 'UGHI (Novel)'
        elif 'rife' in name or vfi == 'rife':
            return 'RIFE (Neural)'
        elif 'adaptive' in name:
            return 'Adaptive (Hybrid)'
        elif 'optical_flow' in name:
            return 'Optical Flow'
        elif 'degraded' in name:
            return 'Baseline'
        else:
            return 'Linear Blend'

    df['category'] = df.apply(get_category, axis=1)
    df['is_novel'] = df['category'].str.contains('Novel')

    return df


def create_comparison_table(df):
    """Create the main comparison table with normalized data."""

    print("\n" + "=" * 70)
    print("NORMALIZED COMPARISON TABLE")
    print("=" * 70)
    print("(PSNR Gain = improvement over 'degraded' baseline on same interval)")
    print()

    # Sort by PSNR gain
    sorted_df = df.sort_values('psnr_gain', ascending=False)

    print(f"{'Method':<25} {'Category':<18} {'PSNR Gain':>10} {'Time Ratio':>10} {'Efficiency':>10}")
    print("-" * 75)

    for _, row in sorted_df.iterrows():
        print(f"{row['name']:<25} {row['category']:<18} {row['psnr_gain']:>+9.2f} dB {row['time_ratio']:>9.1f}x {row['efficiency']:>+9.2f}")

    return sorted_df


def analyze_novel_methods(df):
    """Specific analysis of novel methods vs baselines."""

    print("\n" + "=" * 70)
    print("NOVEL METHODS ANALYSIS")
    print("=" * 70)

    novel = df[df['is_novel']]
    rife = df[df['category'] == 'RIFE (Neural)']
    linear = df[df['category'] == 'Linear Blend']

    if len(novel) == 0:
        print("  No novel methods found in data.")
        return {}

    # Best linear method
    best_linear = linear.loc[linear['psnr_gain'].idxmax()] if len(linear) > 0 else None
    best_rife = rife.loc[rife['psnr_gain'].idxmax()] if len(rife) > 0 else None

    results = {}

    print("\nNovel Methods Performance:")
    print("-" * 50)

    for _, row in novel.iterrows():
        name = row['name']
        results[name] = {
            'psnr_gain': row['psnr_gain'],
            'time_ratio': row['time_ratio'],
            'efficiency': row['efficiency'],
        }

        print(f"\n  {name} ({row['category']}):")
        print(f"    PSNR Gain: {row['psnr_gain']:+.2f} dB over baseline")
        print(f"    Time: {row['time_ratio']:.1f}x baseline")

        if best_linear is not None:
            vs_linear = row['psnr_gain'] - best_linear['psnr_gain']
            print(f"    vs Best Linear ({best_linear['name']}): {vs_linear:+.2f} dB")
            results[name]['vs_best_linear'] = vs_linear

        if best_rife is not None:
            vs_rife = row['psnr_gain'] - best_rife['psnr_gain']
            speed_vs_rife = best_rife['time_ratio'] / row['time_ratio']
            print(f"    vs RIFE: {vs_rife:+.2f} dB, {speed_vs_rife:.1f}x speed")
            results[name]['vs_rife_psnr'] = vs_rife
            results[name]['vs_rife_speed'] = speed_vs_rife

    return results


def generate_paper_table(df):
    """Generate LaTeX table for paper."""

    print("\n" + "=" * 70)
    print("LATEX TABLE FOR PAPER")
    print("=" * 70)

    latex = r"""
\begin{table}[h]
\centering
\caption{VFI Method Comparison (Normalized to Baseline)}
\label{tab:vfi_comparison}
\begin{tabular}{lccccc}
\toprule
Method & Category & PSNR Gain & Time & Efficiency \\
\midrule
"""

    sorted_df = df.sort_values('psnr_gain', ascending=False)

    for _, row in sorted_df.iterrows():
        if row['name'] == 'degraded':
            continue
        latex += f"{row['name'].replace('_', r'\_')} & {row['category']} & "
        latex += f"{row['psnr_gain']:+.2f} dB & {row['time_ratio']:.1f}x & {row['efficiency']:+.2f} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    print(latex)

    # Save to file
    with open(PROJECT_ROOT / 'outputs' / 'paper_table.tex', 'w') as f:
        f.write(latex)

    return latex


def save_normalized_data(df, baselines, novel_analysis):
    """Save normalized data for further analysis."""

    output = {
        'generated_at': datetime.now().isoformat(),
        'methodology': {
            'description': 'PSNR normalized relative to degraded baseline per interval',
            'baseline_method': 'degraded (frame duplication + bicubic)',
            'deduplication': 'median PSNR gain selected for methods with multiple runs',
        },
        'baselines_by_interval': baselines,
        'experiments': df.to_dict('records'),
        'novel_methods_analysis': novel_analysis,
        'summary': {
            'total_methods': len(df),
            'novel_methods': len(df[df['is_novel']]),
            'best_overall': df.loc[df['psnr_gain'].idxmax()]['name'],
            'best_novel': df[df['is_novel']].loc[df[df['is_novel']]['psnr_gain'].idxmax()]['name'] if len(df[df['is_novel']]) > 0 else None,
        }
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved normalized data to: {OUTPUT_FILE}")

    return output


def main():
    """Run the normalization pipeline."""

    # Load and normalize
    df, baselines = load_and_normalize()

    # Deduplicate
    df = deduplicate_methods(df)

    # Categorize
    df = categorize_methods(df)

    # Create comparison table
    df = create_comparison_table(df)

    # Analyze novel methods
    novel_analysis = analyze_novel_methods(df)

    # Generate paper table
    generate_paper_table(df)

    # Save
    output = save_normalized_data(df, baselines, novel_analysis)

    print("\n" + "=" * 70)
    print("NORMALIZATION COMPLETE")
    print("=" * 70)

    return df, output


if __name__ == '__main__':
    df, output = main()
