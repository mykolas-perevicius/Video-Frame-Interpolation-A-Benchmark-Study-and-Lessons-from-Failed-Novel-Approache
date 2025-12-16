#!/usr/bin/env python3
"""Comprehensive statistical analysis for VFI benchmark results.

Computes:
- Per-method aggregates (mean, std, CI, percentiles)
- Paired t-tests between methods
- Effect sizes (Cohen's d)
- Per-frame distributions
- Best visual selection
- Motion-difficulty stratified analysis
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy import stats
from collections import defaultdict

OUTPUTS_DIR = Path(__file__).parent.parent / 'outputs'


def load_results() -> dict:
    """Load experiment results."""
    results_file = OUTPUTS_DIR / 'experiment_results.json'
    if not results_file.exists():
        raise FileNotFoundError("No experiment results found. Run experiments first.")
    with open(results_file) as f:
        return json.load(f)


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float('nan')

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return float('nan')

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cohens_d_ci(d: float, n1: int, n2: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for Cohen's d using approximate formula."""
    if np.isnan(d) or n1 < 2 or n2 < 2:
        return (float('nan'), float('nan'))

    # Approximate standard error of d
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

    # Z-score for confidence level
    z = stats.norm.ppf((1 + confidence) / 2)

    return (d - z * se, d + z * se)


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    if np.isnan(d):
        return "N/A"
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def compute_method_statistics(experiments: List[dict]) -> Dict[str, dict]:
    """Compute per-method aggregate statistics."""

    # Group by method
    by_method = defaultdict(list)
    for exp in experiments:
        method = exp['name']
        if exp.get('vfi_psnr_db', 0) > 0:  # Skip control
            by_method[method].append(exp)

    stats_by_method = {}

    for method, exps in by_method.items():
        n = len(exps)
        vfi_psnrs = [e.get('vfi_psnr_db', 0) for e in exps]
        kf_psnrs = [e.get('keyframe_psnr_db', 0) for e in exps]
        times = [e.get('time_s', 0) for e in exps]
        vfi_ssims = [e.get('vfi_ssim', 0) for e in exps]
        # LPIPS: filter out None values
        vfi_lpips = [e.get('vfi_lpips') for e in exps if e.get('vfi_lpips') is not None]

        # Compute statistics
        if n >= 2:
            vfi_mean, vfi_std = np.mean(vfi_psnrs), np.std(vfi_psnrs, ddof=1)
            time_mean, time_std = np.mean(times), np.std(times, ddof=1)

            # 95% CI using t-distribution
            t_crit = stats.t.ppf(0.975, n - 1)
            vfi_ci = (vfi_mean - t_crit * vfi_std / np.sqrt(n),
                      vfi_mean + t_crit * vfi_std / np.sqrt(n))
            time_ci = (time_mean - t_crit * time_std / np.sqrt(n),
                       time_mean + t_crit * time_std / np.sqrt(n))
        else:
            vfi_mean, vfi_std = vfi_psnrs[0] if vfi_psnrs else 0, 0
            time_mean, time_std = times[0] if times else 0, 0
            vfi_ci = (vfi_mean, vfi_mean)
            time_ci = (time_mean, time_mean)

        # Percentiles
        vfi_percentiles = {
            'p5': float(np.percentile(vfi_psnrs, 5)) if vfi_psnrs else 0,
            'p25': float(np.percentile(vfi_psnrs, 25)) if vfi_psnrs else 0,
            'p50': float(np.percentile(vfi_psnrs, 50)) if vfi_psnrs else 0,
            'p75': float(np.percentile(vfi_psnrs, 75)) if vfi_psnrs else 0,
            'p95': float(np.percentile(vfi_psnrs, 95)) if vfi_psnrs else 0,
        }

        # Best visual (highest VFI PSNR)
        best_idx = np.argmax(vfi_psnrs) if vfi_psnrs else 0
        best_exp = exps[best_idx] if exps else None

        # LPIPS statistics (lower is better, perceptual metric)
        lpips_mean = round(float(np.mean(vfi_lpips)), 4) if vfi_lpips else None
        lpips_std = round(float(np.std(vfi_lpips, ddof=1)), 4) if len(vfi_lpips) >= 2 else None

        stats_by_method[method] = {
            'n': n,
            'vfi_psnr_mean': round(float(vfi_mean), 3),
            'vfi_psnr_std': round(float(vfi_std), 3),
            'vfi_psnr_ci95': [round(vfi_ci[0], 3), round(vfi_ci[1], 3)],
            'vfi_psnr_percentiles': vfi_percentiles,
            'kf_psnr_mean': round(float(np.mean(kf_psnrs)), 3) if kf_psnrs else 0,
            'vfi_ssim_mean': round(float(np.mean(vfi_ssims)), 4) if vfi_ssims else 0,
            'vfi_lpips_mean': lpips_mean,  # Lower is better (perceptual distance)
            'vfi_lpips_std': lpips_std,
            'time_mean': round(float(time_mean), 2),
            'time_std': round(float(time_std), 2),
            'time_ci95': [round(time_ci[0], 2), round(time_ci[1], 2)],
            'best_visual': {
                'interval': best_exp.get('interval', {}) if best_exp else {},
                'vfi_psnr': best_exp.get('vfi_psnr_db', 0) if best_exp else 0,
                'timestamp': best_exp.get('timestamp', '') if best_exp else ''
            }
        }

    return stats_by_method


def compute_pairwise_comparisons(
    experiments: List[dict],
    method_stats: Dict[str, dict]
) -> List[dict]:
    """Compute pairwise statistical comparisons between methods."""

    # Group experiments by interval for paired comparisons
    by_interval = defaultdict(dict)
    for exp in experiments:
        interval_key = str(exp.get('interval', {}).get('start', 'unknown'))
        method = exp['name']
        if exp.get('vfi_psnr_db', 0) > 0:
            by_interval[interval_key][method] = exp.get('vfi_psnr_db', 0)

    methods = list(method_stats.keys())
    comparisons = []

    # Sort by performance for meaningful comparisons
    methods_sorted = sorted(methods, key=lambda m: method_stats[m]['vfi_psnr_mean'], reverse=True)

    # Compare each method to baseline (degraded) and top performer
    baseline = 'degraded'
    top_performer = methods_sorted[0] if methods_sorted else None

    for method in methods_sorted:
        if method == baseline:
            continue

        # Get paired data
        method_vals = []
        baseline_vals = []

        for interval_key, interval_data in by_interval.items():
            if method in interval_data and baseline in interval_data:
                method_vals.append(interval_data[method])
                baseline_vals.append(interval_data[baseline])

        n_paired = len(method_vals)

        if n_paired >= 2:
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(method_vals, baseline_vals)
            d = cohens_d(method_vals, baseline_vals)
            d_ci = cohens_d_ci(d, n_paired, n_paired)
        else:
            # Use independent samples (less powerful)
            method_all = [exp.get('vfi_psnr_db', 0) for exp in experiments
                          if exp['name'] == method and exp.get('vfi_psnr_db', 0) > 0]
            baseline_all = [exp.get('vfi_psnr_db', 0) for exp in experiments
                            if exp['name'] == baseline and exp.get('vfi_psnr_db', 0) > 0]

            if len(method_all) >= 1 and len(baseline_all) >= 1:
                if len(method_all) >= 2 and len(baseline_all) >= 2:
                    t_stat, p_value = stats.ttest_ind(method_all, baseline_all)
                else:
                    t_stat, p_value = float('nan'), float('nan')
                d = cohens_d(method_all, baseline_all)
                d_ci = cohens_d_ci(d, len(method_all), len(baseline_all))
                n_paired = 0  # Indicate unpaired
            else:
                t_stat, p_value, d, d_ci = float('nan'), float('nan'), float('nan'), (float('nan'), float('nan'))

        comparisons.append({
            'method_a': method,
            'method_b': baseline,
            'comparison_type': 'paired' if n_paired >= 2 else 'independent',
            'n_pairs': n_paired,
            'psnr_diff': round(method_stats[method]['vfi_psnr_mean'] - method_stats.get(baseline, {}).get('vfi_psnr_mean', 0), 3),
            'cohens_d': round(d, 3) if not np.isnan(d) else None,
            'cohens_d_ci95': [round(d_ci[0], 3), round(d_ci[1], 3)] if not np.isnan(d_ci[0]) else None,
            'effect_interpretation': interpret_cohens_d(d),
            't_statistic': round(t_stat, 3) if not np.isnan(t_stat) else None,
            'p_value': round(p_value, 4) if not np.isnan(p_value) else None,
            'significant_005': p_value < 0.05 if not np.isnan(p_value) else None,
            'significant_001': p_value < 0.01 if not np.isnan(p_value) else None
        })

    return comparisons


def compute_difficulty_analysis(experiments: List[dict]) -> dict:
    """Analyze performance by motion difficulty."""

    # This would require interval metadata - for now, estimate from results
    # In future, load from interval meta.json files

    return {
        'note': 'Difficulty stratification requires interval metadata',
        'implementation': 'pending'
    }


def generate_summary_markdown(
    method_stats: Dict[str, dict],
    comparisons: List[dict]
) -> str:
    """Generate human-readable summary."""

    lines = [
        "# VFI Benchmark Statistical Analysis",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Method Rankings (by VFI PSNR)",
        "",
        "| Rank | Method | VFI PSNR | LPIPS ↓ | SSIM | Time (s) | n |",
        "|------|--------|----------|---------|------|----------|---|"
    ]

    # Sort by VFI PSNR
    ranked = sorted(method_stats.items(), key=lambda x: x[1]['vfi_psnr_mean'], reverse=True)

    for i, (method, s) in enumerate(ranked, 1):
        lpips = f"{s['vfi_lpips_mean']:.4f}" if s.get('vfi_lpips_mean') is not None else "N/A"
        ssim = f"{s['vfi_ssim_mean']:.4f}" if s.get('vfi_ssim_mean') else "N/A"
        lines.append(f"| {i} | {method} | {s['vfi_psnr_mean']:.2f} dB | {lpips} | {ssim} | {s['time_mean']:.1f} | {s['n']} |")

    lines.extend([
        "",
        "## Statistical Comparisons vs Baseline (degraded)",
        "",
        "| Method | PSNR Diff | Cohen's d | Effect | p-value | Sig. |",
        "|--------|-----------|-----------|--------|---------|------|"
    ])

    for c in comparisons:
        d_str = f"{c['cohens_d']:.2f}" if c['cohens_d'] is not None else "N/A"
        p_str = f"{c['p_value']:.4f}" if c['p_value'] is not None else "N/A"
        sig = "***" if c.get('significant_001') else ("*" if c.get('significant_005') else "")
        lines.append(f"| {c['method_a']} | +{c['psnr_diff']:.2f} dB | {d_str} | {c['effect_interpretation']} | {p_str} | {sig} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Cohen's d**: < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large",
        "- **Significance**: * p < 0.05, *** p < 0.01",
        "- **95% CI**: If intervals don't overlap, difference is likely significant",
        "",
        "## Notes",
        "",
        "- Results may have limited statistical power with small sample sizes",
        "- Run experiments on more intervals for robust conclusions",
        "- Best visual = highest VFI PSNR run (use for demos)",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of VFI benchmark results')
    parser.add_argument('--output', type=str, default='outputs/statistical_report.json',
                        help='Output file for JSON report')
    parser.add_argument('--summary', type=str, default='outputs/statistical_summary.md',
                        help='Output file for markdown summary')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")

    # Load results
    results = load_results()
    experiments = results.get('experiments', [])

    if not experiments:
        print("No experiments found.")
        return

    print(f"Loaded {len(experiments)} experiment runs")

    # Compute statistics
    print("\nComputing per-method statistics...")
    method_stats = compute_method_statistics(experiments)

    print("Computing pairwise comparisons...")
    comparisons = compute_pairwise_comparisons(experiments, method_stats)

    print("Analyzing by difficulty...")
    difficulty_analysis = compute_difficulty_analysis(experiments)

    # Build report
    report = {
        'generated': datetime.now().isoformat(),
        'n_experiments': len(experiments),
        'n_methods': len(method_stats),
        'by_method': method_stats,
        'comparisons': comparisons,
        'difficulty_analysis': difficulty_analysis
    }

    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Generate and save summary
    summary = generate_summary_markdown(method_stats, comparisons)
    summary_path = Path(args.summary)
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Saved: {summary_path}")

    # Print quick summary
    print(f"\n{'='*60}")
    print("QUICK SUMMARY")
    print(f"{'='*60}")

    ranked = sorted(method_stats.items(), key=lambda x: x[1]['vfi_psnr_mean'], reverse=True)
    for i, (method, s) in enumerate(ranked[:5], 1):
        print(f"{i}. {method}: {s['vfi_psnr_mean']:.2f} dB (n={s['n']})")

    sig_count = sum(1 for c in comparisons if c.get('significant_005'))
    print(f"\n{sig_count}/{len(comparisons)} comparisons statistically significant (p < 0.05)")


if __name__ == '__main__':
    main()
