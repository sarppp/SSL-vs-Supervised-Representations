#!/usr/bin/env python3
import os
import json
from pathlib import Path
from collections import defaultdict

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

    # results[(model_family, regime)][label_pct] = {...}
    results = defaultdict(dict)

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

        results[(model_family, regime)][label_pct] = {
            'test_accuracy': test_acc,
            'metrics': metrics,
            'train_config': train_cfg,
            'eval_file': str(ef),
            'train_file': str(tf) if tf.exists() else None,
        }

    return results


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
        lines.append(f'Model: {model_family.upper()} | Regime: {regime}')</n>        label_list = sorted(label_map.keys())
        rep = label_map[label_list[-1]]  # representative config from a label setting
        cfg = rep.get('train_config') or {}
        if cfg:
            lines.append(f"  Epochs: {cfg.get('EPOCHS')}, Batch size: {cfg.get('BATCH_SIZE')}, Effective batch: {cfg.get('EFFECTIVE_BATCH_SIZE')}")
            lines.append(f"  Optimizer: {cfg.get('OPTIMIZER')} lr={cfg.get('LEARNING_RATE')} weight_decay={cfg.get('WEIGHT_DECAY')}")
            lines.append(f"  Scheduler: {cfg.get('SCHEDULER')} warmup={cfg.get('USE_WARMUP')} warmup_epochs={cfg.get('WARMUP_EPOCHS')}")
            lines.append(f"  Mixed precision: {cfg.get('MIXED_PRECISION')} grad_accum_steps={cfg.get('GRADIENT_ACCUM_STEPS')}")
            lines.append(f"  Image size: {tuple(cfg.get('IMAGE_SIZE', (224, 224)))}")
            # Transforms
            tr = cfg.get('TRAIN_TRANSFORMS', {})
            vr = cfg.get('VAL_TEST_TRANSFORMS', {})
            lines.append(f"  Train transforms: resize={tr.get('resize')} augments=[flip, rotation, affine, color_jitter]")
            lines.append(f"  Eval transforms: resize={vr.get('resize')} center-crop/resize, normalize=ImageNet mean/std")
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

    label_points = sorted({lp for _, m in results.items() for lp in m.keys()})
    series_order = [
        ('cnn', 'supervised', 'CNN (supervised)', 'C0'),
        ('vit', 'supervised', 'ViT (supervised)', 'C1'),
        ('dinov2', 'linear_probe', 'DINOv2 (linear probe)', 'C2'),
        ('dinov2', 'fine_tune', 'DINOv2 (fine-tune)', 'C3'),
    ]

    for key_model, key_regime, label, color in series_order:
        if (key_model, key_regime) not in results:
            continue
        ys = []
        xs = []
        for lp in label_points:
            cond = results[(key_model, key_regime)].get(lp)
            if cond is not None:
                xs.append(lp)
                ys.append(cond['test_accuracy'])
        if xs:
            plt.plot(xs, ys, marker='o', label=label, color=color)

    plt.xlabel('Labeled percentage (%)')
    plt.ylabel('Test accuracy (%)')
    plt.title('Label efficiency: accuracy vs labeled fraction')
    plt.xticks(label_points)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = PLOTS_DIR / 'label_efficiency.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    return out_path


def main():
    results = collect_results()
    methods_path = write_methods_txt(results)
    plot_path = plot_label_efficiency(results)
    print(f'Wrote methods summary to: {methods_path}')
    print(f'Wrote label-efficiency plot to: {plot_path}')


if __name__ == '__main__':
    main()


