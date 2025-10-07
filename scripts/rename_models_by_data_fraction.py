#!/usr/bin/env python3
"""
Rename saved model checkpoints to include training data size (fraction or sample count).

Default behavior (dry-run): show planned renames without changing files.
Use --apply to perform renames.

Extras:
- Writes an audit log of renames (old -> new) unless disabled.
- Can update the comparison results JSON with new best_model_path(s) unless disabled.

Name format is configurable; defaults to adding a suffix like '_data5pct'.
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple


def load_results(results_file: Path) -> List[Dict[str, Any]]:
    with open(results_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'model_results' in data:
        return data['model_results']
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported results JSON format: expected list or dict with 'model_results'")


def compute_suffix(result: Dict[str, Any], fmt: str) -> str:
    """Compute suffix using data_fraction/sample_size/accuracy.
    Available placeholders: {data_pct}, {sample_size}, {acc}
    - data_pct: integer percent (e.g., 5, 25, 50)
    - sample_size: integer sample size if available
    - acc: integer-rounded test accuracy (e.g., 83)
    """
    data_fraction = result.get('data_fraction')
    sample_size = result.get('sample_size')
    test_acc = result.get('test_accuracy')

    data_pct_str = None
    try:
        if data_fraction is not None:
            data_pct = int(round(float(data_fraction) * 100))
            data_pct_str = f"{data_pct}"
    except Exception:
        data_pct_str = None

    try:
        ss = int(sample_size) if sample_size is not None else None
    except Exception:
        ss = None

    # Accuracy formatting
    acc_str = None
    try:
        if test_acc is not None:
            acc_int = int(round(float(test_acc)))
            acc_str = f"{acc_int}"
    except Exception:
        acc_str = None

    # Fallbacks for robustness
    if data_pct_str is None and ss is None:
        # If neither data pct nor sample size is available, allow accuracy-only suffix
        if acc_str is None:
            return ""

    return fmt.format(data_pct=data_pct_str if data_pct_str is not None else "NA",
                      sample_size=ss if ss is not None else "NA",
                      acc=acc_str if acc_str is not None else "NA")


def plan_renames(results: List[Dict[str, Any]],
                 name_fmt_suffix: str,
                 only_successful: bool = True) -> List[Tuple[Path, Path]]:
    plans: List[Tuple[Path, Path]] = []
    for r in results:
        if only_successful and not r.get('success', False):
            continue
        best_path = r.get('best_model_path')
        if not best_path:
            continue
        src = Path(best_path)
        if not src.exists():
            continue

        suffix = compute_suffix(r, name_fmt_suffix)
        if not suffix:
            continue

        # Insert suffix before extension
        stem = src.stem
        new_name = f"{stem}{suffix}{src.suffix}"
        dst = src.with_name(new_name)

        if src != dst:
            plans.append((src, dst))
    return plans


def write_audit(plans: List[Tuple[Path, Path]], results_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f"rename_audit_{ts}.json"
    payload = {
        'results_file': str(results_path),
        'timestamp': ts,
        'renames': [{'old': str(src), 'new': str(dst)} for src, dst in plans]
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    return out_path


def update_results_json(results_path: Path, plans: List[Tuple[Path, Path]]) -> bool:
    """Update best_model_path entries in the comparison results JSON in-place.
    Supports both list and {'model_results': [...]} formats.
    """
    if not plans:
        return False
    old_to_new = {str(src): str(dst) for src, dst in plans}
    with open(results_path, 'r') as f:
        data = json.load(f)
    changed = False
    def _update_entry(entry: Dict[str, Any]) -> None:
        nonlocal changed
        path = entry.get('best_model_path')
        if path and path in old_to_new:
            entry['best_model_path'] = old_to_new[path]
            changed = True
    if isinstance(data, dict) and 'model_results' in data and isinstance(data['model_results'], list):
        for entry in data['model_results']:
            if isinstance(entry, dict):
                _update_entry(entry)
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                _update_entry(entry)
    else:
        return False
    if changed:
        with open(results_path, 'w') as f:
            json.dump(data, f, indent=2)
    return changed


def main():
    parser = argparse.ArgumentParser(description="Rename model checkpoints to include training data size")
    parser.add_argument('--results-file', type=str, default=None,
                        help='Path to comparison results JSON (defaults to most recent in outputs/comparison_results)')
    parser.add_argument('--apply', action='store_true', help='Perform renames (otherwise dry-run)')
    parser.add_argument('--suffix-format', type=str, default='_data{data_pct}pct',
                        help='Suffix format using placeholders {data_pct}, {sample_size}, {acc}')
    parser.add_argument('--filter-model', type=str, default=None,
                        help='Optional: only rename for a specific model_type (e.g., cnn, vit, dinov2)')
    parser.add_argument('--no-audit', action='store_true', help='Disable writing rename audit JSON')
    parser.add_argument('--no-update', action='store_true', help='Do not update results JSON with new paths')
    args = parser.parse_args()

    # Resolve results file
    results_path: Path
    if args.results_file:
        results_path = Path(args.results_file)
    else:
        cr_dir = Path('outputs/comparison_results')
        if not cr_dir.exists():
            raise FileNotFoundError("outputs/comparison_results not found and --results-file not provided")
        cand = sorted(cr_dir.glob('comparison_results_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cand:
            cand = list(cr_dir.glob('comparison.json'))
        if not cand:
            raise FileNotFoundError("No comparison results JSON found in outputs/comparison_results")
        results_path = cand[0]

    results = load_results(results_path)

    # Optionally filter
    if args.filter_model:
        results = [r for r in results if r.get('model_type') == args.filter_model]

    plans = plan_renames(results, args.suffix_format)

    if not plans:
        print("No files to rename (missing data_fraction/sample_size or already renamed).")
        return

    print("Planned renames:")
    for src, dst in plans:
        print(f"  {src} -> {dst}")

    if not args.apply:
        print("\nDry-run complete. Re-run with --apply to perform renames.")
        return

    # Apply renames safely
    errors = 0
    done = 0
    for src, dst in plans:
        try:
            if dst.exists():
                print(f"SKIP (exists): {dst}")
                continue
            os.rename(src, dst)
            done += 1
        except Exception as e:
            print(f"ERROR renaming {src} -> {dst}: {e}")
            errors += 1

    print(f"\nRenames completed: {done} succeeded, {errors} errors.")

    # Write audit
    if not args.no_audit:
        audit_dir = Path('outputs') / 'rename_audit'
        audit_path = write_audit(plans, results_path, audit_dir)
        print(f"Audit written: {audit_path}")

    # Update results JSON with new best_model_path values
    if not args.no_update:
        updated = update_results_json(results_path, plans)
        if updated:
            print(f"Updated results JSON with new best_model_path entries: {results_path}")
        else:
            print("Results JSON not changed (no matching best_model_path entries found).")


if __name__ == '__main__':
    main()


