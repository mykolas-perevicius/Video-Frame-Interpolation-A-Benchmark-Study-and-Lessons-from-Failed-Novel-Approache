#!/usr/bin/env python3
"""Clean up experiment results - remove incomplete or duplicate runs."""

import argparse
import json
from pathlib import Path
from collections import defaultdict

OUTPUTS_DIR = Path(__file__).parent.parent / 'outputs'


def load_results():
    """Load experiment results."""
    results_file = OUTPUTS_DIR / 'experiment_results.json'
    if not results_file.exists():
        return None
    with open(results_file) as f:
        return json.load(f)


def save_results(results):
    """Save experiment results."""
    results_file = OUTPUTS_DIR / 'experiment_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def cleanup_results(
    remove_duplicates: bool = True,
    remove_zero_psnr: bool = True,
    keep_best: bool = True,
    dry_run: bool = True
):
    """Clean up experiment results.

    Args:
        remove_duplicates: Remove duplicate runs (same method + interval)
        remove_zero_psnr: Remove experiments with 0 VFI PSNR
        keep_best: When removing duplicates, keep the best run
        dry_run: Show what would be removed without actually removing
    """
    results = load_results()
    if not results:
        print("No results found.")
        return

    experiments = results.get('experiments', [])
    print(f"Loaded {len(experiments)} experiments")

    to_remove = []
    to_keep = []

    # Group by (method, interval_start)
    by_key = defaultdict(list)
    for i, exp in enumerate(experiments):
        key = (exp['name'], exp.get('interval', {}).get('start', 0))
        by_key[key].append((i, exp))

    for key, runs in by_key.items():
        method, interval_start = key

        if len(runs) > 1 and remove_duplicates:
            # Multiple runs for same method+interval
            if keep_best:
                # Keep the one with highest VFI PSNR
                best_idx, best_exp = max(runs, key=lambda x: x[1].get('vfi_psnr_db', 0) or 0)
                for idx, exp in runs:
                    if idx != best_idx:
                        to_remove.append((idx, f"duplicate of {method}@{interval_start}s"))
                to_keep.append((best_idx, best_exp))
            else:
                # Keep first, remove rest
                to_keep.append(runs[0])
                for idx, exp in runs[1:]:
                    to_remove.append((idx, f"duplicate of {method}@{interval_start}s"))
        else:
            to_keep.extend(runs)

    # Remove zero PSNR experiments
    if remove_zero_psnr:
        new_keep = []
        for idx, exp in to_keep:
            vfi_psnr = exp.get('vfi_psnr_db', 0)
            if vfi_psnr == 0 and exp['name'] != 'control':
                to_remove.append((idx, "zero VFI PSNR"))
            else:
                new_keep.append((idx, exp))
        to_keep = new_keep

    # Report
    print(f"\n{'='*60}")
    print(f"CLEANUP SUMMARY")
    print(f"{'='*60}")
    print(f"Keeping: {len(to_keep)} experiments")
    print(f"Removing: {len(to_remove)} experiments")

    if to_remove:
        print(f"\nRemovals:")
        for idx, reason in sorted(to_remove):
            exp = experiments[idx]
            print(f"  [{idx}] {exp['name']} @ {exp.get('interval', {}).get('start', 'N/A')}s - {reason}")

    if dry_run:
        print(f"\n[DRY RUN] No changes made. Use --apply to actually remove.")
    else:
        # Apply changes
        new_experiments = [exp for idx, exp in sorted(to_keep, key=lambda x: x[0])]
        results['experiments'] = new_experiments
        save_results(results)
        print(f"\nSaved {len(new_experiments)} experiments")


def main():
    parser = argparse.ArgumentParser(description='Clean up experiment results')
    parser.add_argument('--apply', action='store_true', help='Actually apply changes (default: dry run)')
    parser.add_argument('--keep-duplicates', action='store_true', help='Do not remove duplicates')
    parser.add_argument('--keep-zero', action='store_true', help='Do not remove zero-PSNR experiments')
    parser.add_argument('--keep-worst', action='store_true', help='Keep first duplicate instead of best')
    args = parser.parse_args()

    cleanup_results(
        remove_duplicates=not args.keep_duplicates,
        remove_zero_psnr=not args.keep_zero,
        keep_best=not args.keep_worst,
        dry_run=not args.apply
    )


if __name__ == '__main__':
    main()
