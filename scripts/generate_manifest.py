#!/usr/bin/env python3
"""
Generate a human-readable manifest (Markdown) by scanning the outputs folder.
Includes key files (checkpoints, logs, results, plots) and, if available,
augments with comparison results metadata (model, regime, data %, accuracy).
"""
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime


def load_results(results_file: Path) -> List[Dict[str, Any]]:
    with open(results_file, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'model_results' in data:
        return data['model_results']
    if isinstance(data, list):
        return data
    return []


def fmt_pct(frac: Any) -> str:
    try:
        return f"{int(float(frac)*100)}%"
    except Exception:
        return "N/A"


def scan_outputs(root: Path) -> Dict[str, List[Path]]:
    cats = {
        'checkpoints': root / 'checkpoints',
        'logs': root / 'logs',
        'evaluation_results': root / 'evaluation_results',
        'training_results': root / 'training_results',
        'plots': root / 'plots',
        'comparison_results': root / 'comparison_results',
    }
    listing: Dict[str, List[Path]] = {k: [] for k in cats}
    for k, p in cats.items():
        if p.exists():
            listing[k] = sorted([q for q in p.rglob('*') if q.is_file()])
    return listing


def pick_latest_results_file(cr_dir: Path) -> Path:
    candidates = list(cr_dir.glob('comparison_results_*.json')) + list(cr_dir.glob('comparison.json'))
    if not candidates:
        return None  # type: ignore
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description='Generate Markdown manifest for outputs directory')
    parser.add_argument('--outputs-dir', type=str, default='outputs')
    parser.add_argument('--results-file', type=str, default=None, help='Optional: specific comparison results JSON')
    parser.add_argument('--out', type=str, default=None, help='Output markdown path (default outputs/manifests/manifest_<ts>.md)')
    args = parser.parse_args()

    root = Path(args.outputs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Outputs directory not found: {root}")

    listing = scan_outputs(root)

    # Load results (optional)
    results_path: Path = None  # type: ignore
    if args.results_file:
        rp = Path(args.results_file)
        results = load_results(rp) if rp.exists() else []
        results_path = rp if rp.exists() else None  # type: ignore
    else:
        cr_dir = root / 'comparison_results'
        rp = pick_latest_results_file(cr_dir) if cr_dir.exists() else None
        results = load_results(rp) if rp else []
        results_path = rp  # type: ignore

    # Build quick index by checkpoint path for metadata enrichment
    meta_by_ckpt: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        ckpt = r.get('best_model_path')
        if ckpt:
            meta_by_ckpt[str(ckpt)] = {
                'model': (r.get('model_type') or '').upper(),
                'regime': r.get('training_regime'),
                'data_pct': fmt_pct(r.get('data_fraction')) if r.get('data_fraction') is not None else None,
                'sample_size': r.get('sample_size'),
                'labels': fmt_pct(r.get('budget_value')) if r.get('budget_value') is not None else None,
                'acc': r.get('test_accuracy'),
                'name': r.get('model_name'),
            }

    # Write markdown
    out_dir = root / 'manifests'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = Path(args.out) if args.out else out_dir / f'manifest_{ts}.md'

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(root))
        except Exception:
            return str(p)

    with open(md_path, 'w') as f:
        f.write(f"# Outputs Manifest ({ts})\n\n")
        if results_path:
            f.write(f"Results source: {results_path}\n\n")

        # Checkpoints with metadata if available
        f.write("## Checkpoints\n\n")
        if listing['checkpoints']:
            f.write('| File | Model | Regime | Data | Labels | Acc (%) |\n')
            f.write('| --- | --- | --- | --- | --- | --- |\n')
            for p in listing['checkpoints']:
                m = meta_by_ckpt.get(str(p)) or {}
                f.write(f"| {rel(p)} | {m.get('model','')} | {m.get('regime','')} | {m.get('data_pct','')} | {m.get('labels','')} | {m.get('acc','')} |\n")
            f.write('\n')
        else:
            f.write('No checkpoints found.\n\n')

        # Logs
        f.write('## Logs\n\n')
        if listing['logs']:
            for p in listing['logs']:
                f.write(f"- {rel(p)}\n")
            f.write('\n')
        else:
            f.write('No logs found.\n\n')

        # Evaluation results
        f.write('## Evaluation Results\n\n')
        if listing['evaluation_results']:
            for p in listing['evaluation_results']:
                f.write(f"- {rel(p)}\n")
            f.write('\n')
        else:
            f.write('No evaluation result files found.\n\n')

        # Training results
        f.write('## Training Results\n\n')
        if listing['training_results']:
            for p in listing['training_results']:
                f.write(f"- {rel(p)}\n")
            f.write('\n')
        else:
            f.write('No training result files found.\n\n')

        # Plots
        f.write('## Plots\n\n')
        if listing['plots']:
            for p in listing['plots']:
                f.write(f"- {rel(p)}\n")
            f.write('\n')
        else:
            f.write('No plots found.\n\n')

        # Comparison results
        f.write('## Comparison Results JSONs\n\n')
        if listing['comparison_results']:
            for p in listing['comparison_results']:
                f.write(f"- {rel(p)}\n")
            f.write('\n')
        else:
            f.write('No comparison results files found.\n\n')

    print(f"Manifest written: {md_path}")


if __name__ == '__main__':
    main()


