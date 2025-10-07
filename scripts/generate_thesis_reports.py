#!/usr/bin/env python3
import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from math import sqrt

import matplotlib.pyplot as plt

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / 'outputs'
EVAL_DIR = OUTPUTS_DIR / 'evaluation_results'
TRAIN_DIR = OUTPUTS_DIR / 'training_results'
PLOTS_DIR = OUTPUTS_DIR / 'plots'
METHODS_TXT = OUTPUTS_DIR / 'thesis_methods.txt'


def safe_read_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def parse_key(filename: str):
    # Filenames look like: evaluation_<model>_<regime>_labelsXXpct.json or training_...
    # Extract components for grouping
    # Example: evaluation_dinov2_vitb14_linear_probe_labels10pct.json
    name = filename.replace('evaluation_', '').replace('training_', '').replace('.json', '')
    parts = name.split('_')
    # Find label percentage
    label_token = [p for p in parts if p.startswith('labels')]
    label_pct = 100
    if label_token:
        try:
            label_pct = int(label_token[0].replace('labels', '').replace('pct', ''))
        except Exception:
            label_pct = 100

    # Determine model family and regime heuristically
    if parts[0] == 'dinov2':
        model_family = 'dinov2'
        regime = 'linear_probe' if 'linear' in parts else ('fine_tune' if 'fine' in parts else 'unknown')
    elif parts[0].startswith('vit'):
        model_family = 'vit'
        regime = 'supervised'
    elif parts[0].startswith('efficientnet'):
        model_family = 'cnn'
        regime = 'supervised'
    else:
        model_family = parts[0]
        regime = 'supervised'

    return model_family, regime, label_pct


def collect_results():
    # Aggregate evaluation metrics and link a representative training config per condition
    eval_files = sorted([p for p in EVAL_DIR.glob('*.json')])
    train_index = {p.name.replace('training_', '').replace('.json', ''): p for p in TRAIN_DIR.glob('*.json')}

    # results[(model_family, regime)][key] = {...}
    # key is label_pct by default; if all labels are 100 and data_fraction is available,
    # downstream code can switch to using data_fraction for x-axis.
    results = defaultdict(dict)

    import re

    def extract_data_fraction_from_name(name: str):
        m = re.search(r"_data(\d+)pct", name)
        if m:
            try:
                return int(m.group(1)) / 100.0
            except Exception:
                return None
        return None

    for ef in eval_files:
        model_family, regime, label_pct = parse_key(ef.name)
        e = safe_read_json(ef)
        test_acc = float(e.get('test_accuracy', 0.0))
        metrics = {
            'precision_macro': e.get('precision_macro'),
            'recall_macro': e.get('recall_macro'),
            'f1_macro': e.get('f1_macro'),
        }

        # Find corresponding training file to get config/methods
        base_key = ef.name.replace('evaluation_', '').replace('.json', '')
        tf = train_index.get(f'{base_key}.json'.replace('evaluation_', 'training_'), None)
        if tf is None:
            tf = TRAIN_DIR / f"training_{base_key}.json"
        train_cfg = None
        if tf.exists():
            t = safe_read_json(tf)
            train_cfg = t.get('session_metadata', {}).get('config', {})
        # Attempt to read data_fraction/sample_size from either eval or training JSONs
        data_fraction = None
        sample_size = None
        # Prefer explicit fields in training config if present
        if isinstance(train_cfg, dict):
            data_fraction = train_cfg.get('data_fraction', data_fraction)
            sample_size = train_cfg.get('sample_size', sample_size)
        # Fallback to evaluation file fields if present
        if data_fraction is None:
            data_fraction = e.get('data_fraction', None)
        if sample_size is None:
            sample_size = e.get('sample_size', None)
        # Final fallback: parse from filename if embedded (e.g., _data5pct)
        if data_fraction is None:
            parsed_df = extract_data_fraction_from_name(ef.name)
            data_fraction = parsed_df if parsed_df is not None else None

        results[(model_family, regime)][label_pct] = {
            'test_accuracy': test_acc,
            'metrics': metrics,
            'train_config': train_cfg,
            'eval_file': str(ef),
            'train_file': str(tf) if tf.exists() else None,
            'data_fraction': data_fraction,
            'sample_size': sample_size,
        }

    return results

def augment_with_comparison_results(results):
    """Augment sparse evaluation-based results with comparison_results JSONs.
    Ensures multiple dataset fractions (e.g., 5/25/50) appear even if some eval JSONs are missing.
    """
    cr_dir = OUTPUTS_DIR / 'comparison_results'
    if not cr_dir.exists():
        return results
    # pick most recent comparison file
    cr_files = sorted(cr_dir.glob('comparison_results_*.json'))
    if not cr_files:
        cr_files = list(cr_dir.glob('comparison.json'))
    if not cr_files:
        return results
    comp_path = max(cr_files, key=lambda p: p.stat().st_mtime)
    try:
        data = safe_read_json(comp_path)
    except Exception:
        return results
    # Normalize to list of model_results
    if isinstance(data, dict) and 'model_results' in data:
        model_results = data['model_results']
    elif isinstance(data, list):
        model_results = data
    else:
        return results

    for r in model_results:
        try:
            model_type = r.get('model_type')
            regime = r.get('training_regime', 'supervised')
            test_acc = float(r.get('test_accuracy', 0) or 0)
            label_budget = r.get('budget_value', 1.0)
            data_fraction = r.get('data_fraction', None)
            # Insert stub entry if missing
            key = (model_type, regime)
            if key not in results:
                results[key] = {}
            # If we are going to pivot by data fraction, stash under label=100 for compatibility
            label_pct = int(round(float(label_budget) * 100)) if label_budget is not None else 100
            entry = results[key].get(label_pct, {})
            # Preserve best (or latest) accuracy per data fraction by storing a small map
            df_map = entry.get('_df_map', {})
            if data_fraction is not None:
                try:
                    df_key = int(round(float(data_fraction) * 100))
                    prev = df_map.get(df_key)
                    if prev is None or test_acc > prev.get('test_accuracy', 0):
                        df_map[df_key] = {
                            'test_accuracy': test_acc,
                            'data_fraction': data_fraction
                        }
                except Exception:
                    pass
            # Also keep a representative top-level accuracy
            if not entry.get('test_accuracy') or test_acc > entry.get('test_accuracy', 0):
                entry['test_accuracy'] = test_acc
            entry['data_fraction'] = entry.get('data_fraction', data_fraction)
            entry['_df_map'] = df_map
            results[key][label_pct] = entry
        except Exception:
            continue
    # Also parse local models/ filenames to backfill fractions (e.g., best_*_dataXXpct_*.pth)
    models_dir = Path('models')
    if models_dir.exists():
        import re
        def infer_key_from_filename(fname: str):
            name = fname.lower()
            if 'efficientnet' in name or 'cnn' in name:
                model_type = 'cnn'
                regime = 'supervised'
            elif 'vit' in name and 'dinov2' not in name:
                model_type = 'vit'
                regime = 'supervised'
            elif 'dinov2' in name:
                model_type = 'dinov2'
                if 'linear_probe' in name:
                    regime = 'linear_probe'
                elif 'fine_tune' in name:
                    regime = 'fine_tune'
                else:
                    regime = 'supervised'
            else:
                model_type, regime = None, None
            return model_type, regime
        for p in models_dir.glob('best_*_data*pct_*.pth'):
            try:
                model_type, regime = infer_key_from_filename(p.name)
                if not model_type or not regime:
                    continue
                m = re.search(r"_data(\d+)pct", p.name)
                if not m:
                    continue
                df_key = int(m.group(1))
                # extract accuracy if present
                acc = None
                if 'acc' in p.name:
                    try:
                        acc = float(p.name.split('acc')[-1].replace('.pth', ''))
                    except Exception:
                        acc = None
                key = (model_type, regime)
                if key not in results:
                    results[key] = {}
                entry = results[key].get(100, {})  # stash under 100% labels compatibility
                df_map = entry.get('_df_map', {})
                prev = df_map.get(df_key)
                if acc is None:
                    acc = entry.get('test_accuracy', 0) or 0
                if prev is None or (acc is not None and acc > prev.get('test_accuracy', 0)):
                    df_map[df_key] = {'test_accuracy': acc if acc is not None else 0.0,
                                      'data_fraction': df_key / 100.0}
                entry['_df_map'] = df_map
                if 'data_fraction' not in entry:
                    entry['data_fraction'] = df_key / 100.0
                if 'test_accuracy' not in entry and acc is not None:
                    entry['test_accuracy'] = acc
                results[key][100] = entry
            except Exception:
                continue
    return results


def format_transform_params(transform_config, transform_name):
    """Format transform parameters into a readable string with actual values.
    
    Supports the specific augmentation techniques used in this project:
    - resize, horizontal_flip, vertical_flip, rotation
    - affine (translate, scale), color_jitter (brightness, contrast, saturation, hue)
    - extra_brightness, extra_contrast, normalize (mean, std)
    """
    if not transform_config:
        return None
    
    if isinstance(transform_config, dict):
        if transform_name == 'rotation':
            degrees = transform_config.get('degrees', 0)
            return f"rotation={degrees}°"
        elif transform_name == 'affine':
            translate = transform_config.get('translate', [0, 0])
            scale = transform_config.get('scale', [1.0, 1.0])
            shear = transform_config.get('shear', 0)
            return f"affine(translate={translate}, scale={scale}, shear={shear})"
        elif transform_name == 'color_jitter':
            brightness = transform_config.get('brightness', 0)
            contrast = transform_config.get('contrast', 0)
            saturation = transform_config.get('saturation', 0)
            hue = transform_config.get('hue', 0)
            return f"color_jitter(brightness={brightness}, contrast={contrast}, saturation={saturation}, hue={hue})"
        elif transform_name == 'normalize':
            mean = transform_config.get('mean', [0.485, 0.456, 0.406])
            std = transform_config.get('std', [0.229, 0.224, 0.225])
            return f"normalize(mean={mean}, std={std})"
        else:
            # Generic dict formatting
            params = ", ".join([f"{k}={v}" for k, v in transform_config.items()])
            return f"{transform_name}({params})"
    else:
        # Simple value (like rotation degrees, vertical_flip probability, etc.)
        if transform_name == 'rotation':
            return f"rotation={transform_config}°"
        elif transform_name == 'vertical_flip':
            return f"vertical_flip={transform_config}"
        elif transform_name == 'resize':
            if isinstance(transform_config, list) and len(transform_config) == 2:
                return f"resize={transform_config[0]}x{transform_config[1]}"
            else:
                return f"resize={transform_config}"
        elif transform_name == 'extra_brightness':
            return f"extra_brightness={transform_config}"
        elif transform_name == 'extra_contrast':
            return f"extra_contrast={transform_config}"
        else:
            return f"{transform_name}={transform_config}"


def write_methods_txt(results):
    lines = []
    lines.append('='*80)
    lines.append('Methods Summary for Thesis')
    lines.append('='*80)
    lines.append('')

    # Pretraining sources
    lines.append('Pretraining sources:')
    lines.append('- CNN (EfficientNet): torchvision weights="DEFAULT" (supervised ImageNet).')
    lines.append('- ViT (timm): timm pretrained=True (supervised ImageNet family).')
    lines.append('- DINOv2: torch.hub facebookresearch/dinov2 (self-supervised).')
    lines.append('')

    # For each condition, summarize compute and evaluation protocol from training config
    for (model_family, regime), label_map in sorted(results.items()):
        lines.append(f'Model: {model_family.upper()} | Regime: {regime}')
        label_list = sorted(label_map.keys())
        rep = label_map[label_list[-1]]  # representative config from a label setting
        cfg = rep.get('train_config') or {}
        if cfg:
            lines.append(f"  Epochs: {cfg.get('EPOCHS')}, Batch size: {cfg.get('BATCH_SIZE')}, Effective batch: {cfg.get('EFFECTIVE_BATCH_SIZE')}")
            lines.append(f"  Optimizer: {cfg.get('OPTIMIZER')} lr={cfg.get('LEARNING_RATE')} weight_decay={cfg.get('WEIGHT_DECAY')}")
            lines.append(f"  Scheduler: {cfg.get('SCHEDULER')} warmup={cfg.get('USE_WARMUP')} warmup_epochs={cfg.get('WARMUP_EPOCHS')}")
            lines.append(f"  Mixed precision: {cfg.get('MIXED_PRECISION')} grad_accum_steps={cfg.get('GRADIENT_ACCUM_STEPS')}")
            lines.append(f"  Image size: {tuple(cfg.get('IMAGE_SIZE', (224, 224)))}")
            # Transforms with detailed parameters
            tr = cfg.get('TRAIN_TRANSFORMS', {})
            vr = cfg.get('VAL_TEST_TRANSFORMS', {})
            
            # Build detailed train transforms description
            train_transforms = []
            
            # Transform keys actually used in this project
            transform_keys = [
                'resize', 'horizontal_flip', 'vertical_flip', 'rotation', 'affine', 
                'color_jitter', 'extra_brightness', 'extra_contrast', 'normalize'
            ]
            
            for key in transform_keys:
                if tr.get(key):
                    if key == 'horizontal_flip':
                        train_transforms.append("horizontal_flip=True")
                    else:
                        formatted = format_transform_params(tr.get(key), key)
                        if formatted:
                            train_transforms.append(formatted)
            
            # Build detailed validation transforms description
            val_transforms = []
            for key in ['resize', 'normalize']:
                if vr.get(key):
                    formatted = format_transform_params(vr.get(key), key)
                    if formatted:
                        val_transforms.append(formatted)
            
            # Format the transform descriptions
            train_desc = ", ".join(train_transforms) if train_transforms else "basic transforms"
            val_desc = ", ".join(val_transforms) if val_transforms else "basic transforms"
            
            lines.append(f"  Train transforms: {train_desc}")
            lines.append(f"  Eval transforms: {val_desc}")
        # Freeze status by regime
        if model_family == 'dinov2':
            if regime == 'linear_probe':
                lines.append('  Backbone: frozen (linear probe).')
            elif regime == 'fine_tune':
                lines.append('  Backbone: frozen initially, unfrozen after epoch 3.')
        else:
            lines.append('  Backbone: trainable (supervised from scratch or pretrained fine-tune).')
        lines.append('')

    # Include metrics mention
    lines.append('Reported metrics: accuracy, macro precision/recall/F1 (per evaluation JSONs).')
    lines.append('Evaluation protocol: single center-resize/normalize on held-out test split.')
    lines.append('')

    METHODS_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(METHODS_TXT, 'w') as f:
        f.write('\n'.join(lines))
    return METHODS_TXT


def plot_label_efficiency(results):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))

    # Merge in comparison_results to ensure we have all dataset fractions
    results = augment_with_comparison_results(results)

    # Detect if dataset fractions are present; don't require all labels==100 (mixed histories still pivot to data)
    has_any = any(results.values())
    entries = []
    df_candidates = set()
    for (_, _reg), m in results.items():
        for lp, cond in m.items():
            entries.append((lp, cond))
            # collect from explicit field
            if cond.get('data_fraction') is not None:
                try:
                    df_candidates.add(int(round(float(cond.get('data_fraction')) * 100)))
                except Exception:
                    pass
            # collect from consolidated _df_map
            df_map = cond.get('_df_map')
            if isinstance(df_map, dict):
                for k in df_map.keys():
                    try:
                        df_candidates.add(int(k))
                    except Exception:
                        continue
    any_data_fraction = len(df_candidates) > 0
    use_data_fraction = any_data_fraction

    if use_data_fraction:
        # Build x-axis from consolidated set (df_candidates) but restrict to canonical fractions
        target_fractions = {5, 25, 50}
        data_points = sorted((set(df_candidates) & target_fractions) or df_candidates)
        xlabel = 'Dataset fraction (%)'
        title = 'Data efficiency: accuracy vs dataset fraction'
    else:
        data_points = sorted({lp for (lp, _cond) in entries})
        xlabel = 'Labeled percentage (%)'
        title = 'Label efficiency: accuracy vs labeled fraction'

    series_order = [
        ('cnn', 'supervised', 'CNN (supervised)', 'C0'),
        ('vit', 'supervised', 'ViT (supervised)', 'C1'),
        ('dinov2', 'linear_probe', 'DINOv2 (linear probe)', 'C2'),
        ('dinov2', 'fine_tune', 'DINOv2 (fine-tune)', 'C3'),
    ]

    for key_model, key_regime, label, color in series_order:
        if (key_model, key_regime) not in results:
            continue
        # Collapse to one point per dataset fraction to avoid duplicates
        best_by_dp = {}
        if use_data_fraction:
            # First, fold in consolidated maps across all label_pct entries
            for (_lp, cond) in results[(key_model, key_regime)].items():
                df_map = cond.get('_df_map')
                if isinstance(df_map, dict):
                    for dp_key, rec in df_map.items():
                        try:
                            dp_int = int(dp_key)
                            acc = float(rec.get('test_accuracy', 0) or 0)
                            if dp_int not in best_by_dp or acc > best_by_dp[dp_int]:
                                best_by_dp[dp_int] = acc
                        except Exception:
                            continue
            # Then, consider single-entry data_fraction fields
            for (_lp, cond) in results[(key_model, key_regime)].items():
                df = cond.get('data_fraction')
                if df is None:
                    continue
                try:
                    dp_int = int(round(float(df) * 100))
                    acc = float(cond.get('test_accuracy', 0) or 0)
                    if dp_int not in best_by_dp or acc > best_by_dp[dp_int]:
                        best_by_dp[dp_int] = acc
                except Exception:
                    continue
            xs = sorted([dp for dp in set(data_points) if dp in best_by_dp])
            ys = [best_by_dp[dp] for dp in xs]
        else:
            # Label-percentage mode: pick a single accuracy per label budget
            xs, ys = [], []
            for dp in sorted(set(data_points)):
                cond = results[(key_model, key_regime)].get(dp)
                if cond is not None:
                    xs.append(dp)
                    ys.append(cond.get('test_accuracy'))
        if xs:
            plt.scatter(xs, ys, label=label, color=color, s=60, edgecolors='white', linewidths=1.0)
            # Debug summary per-series
            try:
                series_counts = {int(x): 1 for x in xs}
                print(f"[label_efficiency] {label}: points per fraction = {sorted(series_counts.keys())}")
            except Exception:
                pass

    plt.xlabel(xlabel)
    plt.ylabel('Test accuracy (%)')
    plt.title(title)
    plt.xticks(data_points)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = PLOTS_DIR / 'label_efficiency.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    # Global debug summary
    try:
        print(f"[label_efficiency] x-axis fractions: {data_points}")
    except Exception:
        pass
    return out_path


def bootstrap_ci_accuracy(preds, labels, n_boot=5000, alpha=0.05, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    n = len(labels)
    idx = np.arange(n)
    accs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(idx, size=n, replace=True)
        accs[i] = (preds[sample] == labels[sample]).mean() * 100.0
    lower = np.percentile(accs, 100 * (alpha / 2))
    upper = np.percentile(accs, 100 * (1 - alpha / 2))
    return accs.mean(), (lower, upper)


def mcnemar_table(preds_a, preds_b, labels):
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    labels = np.asarray(labels)
    correct_a = preds_a == labels
    correct_b = preds_b == labels
    b01 = np.logical_and(~correct_a, correct_b).sum()  # A wrong, B right
    b10 = np.logical_and(correct_a, ~correct_b).sum()  # A right, B wrong
    return b01, b10


def mcnemar_exact_p(b01, b10):
    # Exact binomial test for symmetric null: p = sum_{k>=max(b01,b10)} C(n,k) 0.5^n
    from math import comb
    n = b01 + b10
    if n == 0:
        return 1.0
    k = max(b01, b10)
    p = sum(comb(n, i) for i in range(k, n + 1)) * (0.5 ** n)
    return p


def ci_and_significance(results):
    # For each condition, compute bootstrap CI using stored predictions
    # Also compute McNemar between DINOv2 linear-probe vs CNN/VIT for shared label budgets
    stats_lines = []
    stats_lines.append('='*80)
    stats_lines.append('Uncertainty and Significance (Bootstrap CI and McNemar)')
    stats_lines.append('='*80)
    pairs = [('dinov2', 'linear_probe'), ('cnn', 'supervised'), ('vit', 'supervised')]
    # Build quick index from eval files
    def key(fam, reg):
        return (fam, reg)

    # Bootstrap CIs
    for (fam, reg), label_map in sorted(results.items()):
        for lp, entry in sorted(label_map.items()):
            ef = Path(entry['eval_file'])
            data = safe_read_json(ef)
            preds = data.get('predictions')
            labels = data.get('true_labels')
            if preds is None or labels is None:
                stats_lines.append(f'{fam.upper()} {reg} @ {lp}%: predictions unavailable; CI skipped')
                continue
            mean_acc, (lo, hi) = bootstrap_ci_accuracy(preds, labels)
            stats_lines.append(f'{fam.upper()} {reg} @ {lp}%: {mean_acc:.2f}% (95% CI {lo:.2f}, {hi:.2f})')

    # McNemar tests at shared budgets
    shared_budgets = sorted(set.intersection(*[set(results.get(key(*p), {}).keys()) for p in pairs if key(*p) in results]))
    for lp in shared_budgets:
        # Compare DINOv2 vs CNN
        if key('dinov2', 'linear_probe') in results and key('cnn', 'supervised') in results:
            a = safe_read_json(Path(results[key('dinov2', 'linear_probe')][lp]['eval_file']))
            b = safe_read_json(Path(results[key('cnn', 'supervised')][lp]['eval_file']))
            b01, b10 = mcnemar_table(a['predictions'], b['predictions'], a['true_labels'])
            p = mcnemar_exact_p(b01, b10)
            stats_lines.append(f'McNemar (DINOv2 vs CNN) @ {lp}%: b01={b01}, b10={b10}, p={p:.4f}')
        # Compare DINOv2 vs ViT
        if key('dinov2', 'linear_probe') in results and key('vit', 'supervised') in results and lp in results[key('vit', 'supervised')]:
            a = safe_read_json(Path(results[key('dinov2', 'linear_probe')][lp]['eval_file']))
            b = safe_read_json(Path(results[key('vit', 'supervised')][lp]['eval_file']))
            b01, b10 = mcnemar_table(a['predictions'], b['predictions'], a['true_labels'])
            p = mcnemar_exact_p(b01, b10)
            stats_lines.append(f'McNemar (DINOv2 vs ViT) @ {lp}%: b01={b01}, b10={b10}, p={p:.4f}')

    out = OUTPUTS_DIR / 'thesis_stats.txt'
    with open(out, 'w') as f:
        f.write('\n'.join(stats_lines))
    return out


def main():
    results = collect_results()
    methods_path = write_methods_txt(results)
    plot_path = plot_label_efficiency(results)
    stats_path = ci_and_significance(results)
    print(f'Wrote methods summary to: {methods_path}')
    print(f'Wrote label-efficiency plot to: {plot_path}')
    print(f'Wrote CI and significance stats to: {stats_path}')


if __name__ == '__main__':
    main()


