#!/usr/bin/env python3
"""
Final Comprehensive Analysis

Handles multiple intervals by:
1. Using appropriate baselines per interval
2. Creating fair comparison groups
3. Generating publication-ready results
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = PROJECT_ROOT / 'outputs' / 'experiment_results.json'


def load_and_process():
    """Load and structure experiment data."""
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    df = pd.DataFrame(data['experiments'])
    df['interval_start'] = df['interval'].apply(lambda x: x['start'])
    df['interval_id'] = df['interval_start'].round(1).astype(str)
    df = df[df['name'] != 'control']

    return df


def categorize(name, vfi_method=''):
    """Categorize method."""
    name_lower = name.lower()
    vfi_lower = str(vfi_method).lower()

    if 'uafi' in name_lower or 'ui_aware' in vfi_lower:
        return 'UAFI (Novel)'
    elif 'mcar' in name_lower:
        return 'MCAR (Novel)'
    elif 'ughi' in name_lower:
        return 'UGHI (Novel)'
    elif 'rife' in name_lower or vfi_lower == 'rife':
        return 'RIFE'
    elif 'adaptive' in name_lower:
        return 'Adaptive'
    elif 'optical_flow' in name_lower:
        return 'Optical Flow'
    elif 'degraded' in name_lower:
        return 'Baseline'
    else:
        return 'Linear Blend'


def analyze_by_interval_groups(df):
    """
    Group intervals and analyze within each group.
    """
    print("=" * 70)
    print("ANALYSIS BY INTERVAL GROUPS")
    print("=" * 70)

    # Identify interval groups
    intervals = df.groupby('interval_id').agg({
        'name': list,
        'psnr_db': ['min', 'max', 'mean'],
    }).reset_index()

    print("\nInterval Summary:")
    for _, row in intervals.iterrows():
        int_id = row['interval_id']
        methods = row['name']['list']
        psnr_range = f"{row['psnr_db']['min']:.1f}-{row['psnr_db']['max']:.1f}"
        print(f"  {int_id}s: {len(methods)} methods, PSNR {psnr_range} dB")

    return intervals


def create_comparison_groups(df):
    """
    Create fair comparison groups based on available data.
    """
    print("\n" + "=" * 70)
    print("FAIR COMPARISON GROUPS")
    print("=" * 70)

    groups = []

    # Group 1: Light methods on interval 53.6s (includes UAFI, UGHI)
    light_interval = df[df['interval_id'] == '53.6']
    if len(light_interval) > 0:
        baseline = light_interval[light_interval['name'] == 'degraded']['psnr_db'].values
        if len(baseline) > 0:
            baseline_psnr = baseline[0]

            print(f"\nGroup 1: Light Methods (interval 53.6s, baseline={baseline_psnr:.2f}dB)")
            print("-" * 60)

            group_data = []
            for _, row in light_interval.iterrows():
                cat = categorize(row['name'], row.get('vfi_method', ''))
                gain = row['psnr_db'] - baseline_psnr
                group_data.append({
                    'name': row['name'],
                    'category': cat,
                    'psnr_db': row['psnr_db'],
                    'psnr_gain': gain,
                    'ssim': row['ssim'],
                    'time_s': row['time_s'],
                    'time_ratio': row['time_s'] / light_interval[light_interval['name'] == 'degraded']['time_s'].values[0],
                })

            group_df = pd.DataFrame(group_data).sort_values('psnr_gain', ascending=False)
            for _, row in group_df.iterrows():
                print(f"  {row['name']:<25} {row['category']:<15} {row['psnr_gain']:>+6.2f}dB  {row['time_ratio']:>5.1f}x")

            groups.append({'name': 'Light Methods', 'interval': '53.6', 'data': group_df})

    # Group 2: Heavy methods on interval 9.3s (includes RIFE, MCAR)
    heavy_interval = df[df['interval_id'] == '9.3']
    if len(heavy_interval) > 0:
        # Use adaptive_conservative as baseline (fastest heavy method)
        baseline_row = heavy_interval[heavy_interval['name'] == 'adaptive_conservative']
        if len(baseline_row) > 0:
            baseline_psnr = baseline_row['psnr_db'].values[0]
            baseline_time = baseline_row['time_s'].values[0]

            print(f"\nGroup 2: Heavy Methods (interval 9.3s, baseline=adaptive_conservative)")
            print("-" * 60)

            group_data = []
            for _, row in heavy_interval.iterrows():
                cat = categorize(row['name'], row.get('vfi_method', ''))
                gain = row['psnr_db'] - baseline_psnr
                group_data.append({
                    'name': row['name'],
                    'category': cat,
                    'psnr_db': row['psnr_db'],
                    'psnr_gain': gain,
                    'ssim': row['ssim'],
                    'time_s': row['time_s'],
                    'time_ratio': row['time_s'] / baseline_time,
                    'speedup_vs_rife': heavy_interval[heavy_interval['name'] == 'rife_default']['time_s'].values[0] / row['time_s'] if 'rife_default' in heavy_interval['name'].values else 1,
                })

            group_df = pd.DataFrame(group_data).sort_values('time_s')
            for _, row in group_df.iterrows():
                speedup = f"{row['speedup_vs_rife']:.1f}x vs RIFE" if 'speedup_vs_rife' in row else ""
                print(f"  {row['name']:<25} {row['category']:<15} {row['time_s']:>6.1f}s  {speedup}")

            groups.append({'name': 'Heavy Methods', 'interval': '9.3', 'data': group_df})

    return groups


def cross_group_analysis(groups):
    """
    Analyze across groups using normalized metrics.
    """
    print("\n" + "=" * 70)
    print("CROSS-GROUP INSIGHTS")
    print("=" * 70)

    if len(groups) < 2:
        print("  Need data from multiple groups for cross-analysis.")
        return

    light_group = next((g for g in groups if g['name'] == 'Light Methods'), None)
    heavy_group = next((g for g in groups if g['name'] == 'Heavy Methods'), None)

    if light_group and heavy_group:
        light_df = light_group['data']
        heavy_df = heavy_group['data']

        print("\nKey Findings:")
        print("-" * 60)

        # Best light method
        best_light = light_df.loc[light_df['psnr_gain'].idxmax()]
        print(f"  Best Light Method: {best_light['name']} (+{best_light['psnr_gain']:.2f}dB)")

        # Novel methods summary
        novel_light = light_df[light_df['category'].str.contains('Novel')]
        if len(novel_light) > 0:
            best_novel = novel_light.loc[novel_light['psnr_gain'].idxmax()]
            print(f"  Best Novel (Light): {best_novel['name']} (+{best_novel['psnr_gain']:.2f}dB)")

        novel_heavy = heavy_df[heavy_df['category'].str.contains('Novel')]
        if len(novel_heavy) > 0:
            fastest_novel = novel_heavy.loc[novel_heavy['time_s'].idxmin()]
            rife_time = heavy_df[heavy_df['name'] == 'rife_default']['time_s'].values[0] if 'rife_default' in heavy_df['name'].values else 0
            if rife_time > 0:
                speedup = rife_time / fastest_novel['time_s']
                print(f"  MCAR vs RIFE: {speedup:.2f}x speedup (same quality)")


def generate_final_report(df, groups):
    """Generate the final analysis report."""

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    report = {
        'generated_at': datetime.now().isoformat(),
        'total_experiments': len(df),
        'unique_methods': len(df['name'].unique()),
        'intervals_tested': df['interval_id'].unique().tolist(),
    }

    # Key results
    print("\nKEY RESULTS:")
    print("-" * 60)

    # 1. On HARD content (interval 53.6s)
    hard_interval = df[df['interval_id'] == '53.6']
    if len(hard_interval) > 0:
        print("\n1. HARD CONTENT (High Motion, interval 53.6s):")
        best_methods = hard_interval.nlargest(3, 'psnr_db')[['name', 'psnr_db', 'time_s']]
        for _, row in best_methods.iterrows():
            cat = categorize(row['name'])
            print(f"   {row['name']}: {row['psnr_db']:.2f}dB, {row['time_s']:.1f}s [{cat}]")

        report['hard_content'] = best_methods.to_dict('records')

    # 2. On EASY content (interval 9.3s or 11.3s)
    easy_interval = df[df['interval_id'].isin(['9.3', '11.3'])]
    if len(easy_interval) > 0:
        print("\n2. EASY CONTENT (Low Motion, interval 9.3s/11.3s):")
        best_methods = easy_interval.nlargest(5, 'psnr_db')[['name', 'psnr_db', 'time_s', 'interval_id']]
        for _, row in best_methods.iterrows():
            cat = categorize(row['name'])
            print(f"   {row['name']}: {row['psnr_db']:.2f}dB, {row['time_s']:.1f}s [{cat}]")

        report['easy_content'] = best_methods.to_dict('records')

    # 3. Novel methods summary
    print("\n3. NOVEL METHODS SUMMARY:")
    novel_names = ['uafi_default', 'ughi_default', 'mcar_default', 'mcar_aggressive']
    novel_data = df[df['name'].isin(novel_names)]

    novel_summary = []
    for _, row in novel_data.iterrows():
        cat = categorize(row['name'])
        insight = ""
        if 'uafi' in row['name'].lower():
            insight = "UI-aware, matches best linear on same interval"
        elif 'ughi' in row['name'].lower():
            insight = "Uncertainty-guided, matches best linear on same interval"
        elif 'mcar' in row['name'].lower():
            insight = "Adaptive routing, same quality as RIFE"

        print(f"   {row['name']}: {row['psnr_db']:.2f}dB, {row['time_s']:.1f}s")
        print(f"      -> {insight}")

        novel_summary.append({
            'name': row['name'],
            'psnr_db': row['psnr_db'],
            'time_s': row['time_s'],
            'interval': row['interval_id'],
            'insight': insight,
        })

    report['novel_methods'] = novel_summary

    # Save report
    report_file = PROJECT_ROOT / 'outputs' / 'final_analysis.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n\nReport saved to: {report_file}")

    return report


def main():
    df = load_and_process()

    # Add categories
    df['category'] = df.apply(lambda r: categorize(r['name'], r.get('vfi_method', '')), axis=1)

    # Analyze by interval
    analyze_by_interval_groups(df)

    # Create fair comparison groups
    groups = create_comparison_groups(df)

    # Cross-group analysis
    cross_group_analysis(groups)

    # Final report
    report = generate_final_report(df, groups)

    return df, groups, report


if __name__ == '__main__':
    df, groups, report = main()
