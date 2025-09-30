#!/usr/bin/env python3
"""
🎓 Thesis Analysis & Visualization Suite
Generates publication-quality tables and figures from experimental results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_comparison_results(results_file=None):
    """Load comparison results from JSON"""
    if results_file is None:
        # Find most recent results file
        results_files = glob.glob('outputs/comparison_results/comparison_results_*.json')
        if not results_files:
            results_files = glob.glob('outputs/comparison_results/comparison.json')
        if not results_files:
            raise FileNotFoundError("No comparison results found!")
        results_file = max(results_files, key=lambda x: Path(x).stat().st_mtime)
    
    print(f"📂 Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)
    # Handle two formats:
    # 1) New: dict with key 'model_results' (list of results)
    # 2) Legacy: list of results directly
    if isinstance(data, dict) and 'model_results' in data:
        return data['model_results']
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported results JSON format: expected list or dict with 'model_results'")

def create_summary_table(results):
    """Create summary table with all metrics for thesis"""
    
    summary_data = []
    
    for result in results:
        if not result.get('success', False):
            continue
        
        row = {
            'Model': result['model_type'].upper(),
            'Regime': result.get('training_regime', 'N/A'),
            'Label Budget': f"{int(result.get('budget_value', 0)*100)}%" if result.get('budget_mode') == 'percentage' else result.get('budget_value', 'N/A'),
            'Test Acc (%)': f"{result.get('test_accuracy', 0):.2f}",
            'Precision': f"{result.get('precision_macro', 0):.3f}",
            'Recall': f"{result.get('recall_macro', 0):.3f}",
            'F1-Score': f"{result.get('f1_macro', 0):.3f}",
            'MCC': f"{result.get('matthews_corrcoef', 0):.3f}",
            'ROC-AUC': f"{result.get('roc_auc_ovr', 0):.3f}" if result.get('roc_auc_ovr') else 'N/A',
            'Train Samples': result.get('labeled_samples', 0),
            'Epochs': result.get('epochs', 0),
            'Time (min)': f"{result.get('time', 0)/60:.1f}",
            'Time/Epoch (s)': f"{result.get('time_per_epoch', 0):.1f}",
            'Images/sec': f"{result.get('images_per_sec', 0):.1f}",
            'GPU Preset': result.get('gpu_preset', 'N/A')
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    return df

def save_latex_table(df, filename='thesis_results_table.tex'):
    """Save DataFrame as LaTeX table for thesis"""
    
    latex_str = df.to_latex(
        index=False,
        caption='Experimental Results: Model Performance Across Label Budgets',
        label='tab:results',
        column_format='l' + 'c' * (len(df.columns) - 1),
        escape=False
    )
    
    output_path = Path('outputs/plots') / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    print(f"📝 LaTeX table saved: {output_path}")
    return output_path

def plot_label_efficiency_curve(results, output_dir='outputs/plots'):
    """
    🎓 THESIS FIGURE: Label efficiency curves
    Shows how accuracy scales with % of labeled data
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Organize data by model/regime
    data = {}
    for result in results:
        if not result.get('success', False):
            continue
        
        model_regime = f"{result['model_type'].upper()} ({result.get('training_regime', 'N/A')})"
        budget = result.get('budget_value', 0) * 100  # Convert to percentage
        accuracy = result.get('test_accuracy', 0)
        
        if model_regime not in data:
            data[model_regime] = {'budgets': [], 'accuracies': []}
        
        data[model_regime]['budgets'].append(budget)
        data[model_regime]['accuracies'].append(accuracy)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    for model_regime, values in data.items():
        # Sort by budget
        sorted_pairs = sorted(zip(values['budgets'], values['accuracies']))
        budgets, accs = zip(*sorted_pairs)
        
        plt.plot(budgets, accs, marker='o', linewidth=2, markersize=8, label=model_regime)
    
    plt.xlabel('Labeled Data (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Label Efficiency: Accuracy vs. Labeled Data Percentage', fontsize=14, pad=15)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save both formats
    png_path = Path(output_dir) / 'label_efficiency_curve.png'
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Label efficiency plot saved:")
    print(f"   PNG: {png_path}")
    
    return png_path

def plot_model_comparison_bar(results, output_dir='outputs/plots'):
    """
    🎓 THESIS FIGURE: Model comparison bar chart
    Show available metrics across models and label budgets.
    - Always plots Accuracy
    - Plots Precision/Recall/F1 only if present in results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build rows for all label budgets
    rows = []
    has_precision = False
    has_recall = False
    has_f1 = False

    for r in results:
        if not r.get('success', False):
            continue
        label_pct = int(r.get('budget_value', 0) * 100)
        model_regime = f"{r['model_type'].upper()}\n({r.get('training_regime', 'N/A')})\n{label_pct}% labels"

        precision = r.get('precision_macro')
        recall = r.get('recall_macro')
        f1 = r.get('f1_macro')

        has_precision = has_precision or (precision is not None)
        has_recall = has_recall or (recall is not None)
        has_f1 = has_f1 or (f1 is not None)

        rows.append({
            'ModelConfig': model_regime,
            'Accuracy': r.get('test_accuracy', 0),
            'Precision': precision * 100 if precision is not None else np.nan,
            'Recall': recall * 100 if recall is not None else np.nan,
            'F1-Score': f1 * 100 if f1 is not None else np.nan,
        })

    if not rows:
        print("⚠️ No successful results to plot.")
        return None

    df = pd.DataFrame(rows).set_index('ModelConfig')

    # Determine metrics to plot dynamically
    metrics = ['Accuracy']
    colors = ['#3498db']
    if has_precision:
        metrics.append('Precision')
        colors.append('#e74c3c')
    if has_recall:
        metrics.append('Recall')
        colors.append('#2ecc71')
    if has_f1:
        metrics.append('F1-Score')
        colors.append('#f39c12')

    # Grouped bar plot
    fig, ax = plt.subplots(figsize=(14, max(6, 0.45 * len(df.index))))
    x = np.arange(len(df.index))
    width = min(0.8 / max(1, len(metrics)), 0.25)

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = df[metric].values
        ax.bar(x + i * width, values, width, label=metric, color=color)

    ax.set_xlabel('Model Configuration / Label Budget', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Comparison across Label Budgets', fontsize=14, pad=15)
    ax.set_xticks(x + (len(metrics) - 1) * width / 2)
    ax.set_xticklabels(df.index, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    png_path = Path(output_dir) / 'model_comparison_bar.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Model comparison plot saved:")
    print(f"   PNG: {png_path}")

    return png_path

def compute_label_efficiency_metrics(results, low_budget_threshold=0.1):
    """Compute AULC (<= low_budget_threshold) and label savings numbers."""
    # Organize accuracy vs budget by (model, regime)
    curves = {}
    for r in results:
        if not r.get('success', False):
            continue
        key = f"{r['model_type'].upper()} ({r.get('training_regime', 'N/A')})"
        curves.setdefault(key, []).append((r.get('budget_value', 0.0), r.get('test_accuracy', 0.0)))
    for k in curves:
        curves[k] = sorted(curves[k])

    import numpy as np
    def aulc(points, max_budget=0.1):
        xs = [x for x,_ in points if x <= max_budget]
        ys = [y for x,y in points if x <= max_budget]
        if len(xs) < 2:
            return None
        return float(np.trapz(ys, xs))

    # Compute for all keys
    aulc_by_key = {k: aulc(v, low_budget_threshold) for k, v in curves.items()}
    return aulc_by_key

def append_efficiency_summary_to_executive(results, output_file='outputs/executive_summary.txt', low_budget_threshold=0.1):
    """Append AULC summary to the existing executive summary file."""
    aulc_by_key = compute_label_efficiency_metrics(results, low_budget_threshold)
    lines = ["\nAULC (<= {:.0f}% labels):".format(low_budget_threshold*100)]
    for k, v in sorted(aulc_by_key.items()):
        if v is not None:
            lines.append(f"- {k}: {v:.2f}")
    try:
        with open(output_file, 'a') as f:
            f.write("\n" + "\n".join(lines) + "\n")
        print("📝 Added AULC summary to:", output_file)
    except Exception as e:
        print("⚠️ Could not append AULC summary:", e)

def generate_executive_summary(results, output_file='outputs/executive_summary.txt'):
    """Generate executive summary for thesis"""
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Find best model
    successful = [r for r in results if r.get('success', False)]
    if not successful:
        print(" No successful results to summarize")
        return
    
    best = max(successful, key=lambda x: x.get('test_accuracy', 0))
    
    # Calculate statistics
    total_experiments = len(results)
    successful_experiments = len(successful)
    
    # Get results by budget
    by_budget = {}
    for r in successful:
        budget = r.get('budget_value', 0)
        if budget not in by_budget:
            by_budget[budget] = []
        by_budget[budget].append(r.get('test_accuracy', 0))
    
    # Generate summary
    summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║             SUMMARY OF EXPERIMENTS            ║
╚══════════════════════════════════════════════════════════════════╝

📊 EXPERIMENT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Experiments:     {total_experiments}
Successful:            {successful_experiments}
Failed:                {total_experiments - successful_experiments}

🏆 BEST PERFORMING MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model:                 {best['model_type'].upper()}
Training Regime:       {best.get('training_regime', 'N/A')}
Label Budget:          {int(best.get('budget_value', 0)*100)}%
Test Accuracy:         {best.get('test_accuracy', 0):.2f}%
Precision (macro):     {best.get('precision_macro', 0):.3f}
Recall (macro):        {best.get('recall_macro', 0):.3f}
F1-Score (macro):      {best.get('f1_macro', 0):.3f}
Matthews Corr. Coef:   {best.get('matthews_corrcoef', 0):.3f}

📈 PERFORMANCE BY LABEL BUDGET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    for budget in sorted(by_budget.keys()):
        accs = by_budget[budget]
        avg_acc = np.mean(accs)
        max_acc = np.max(accs)
        summary += f"{int(budget*100):3d}% labels: Avg={avg_acc:5.2f}% | Best={max_acc:5.2f}%\n"
    
    summary += f"""
 KEY FINDINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Best model: {best['model_type'].upper()} with {best.get('training_regime', 'N/A')} 
   achieved {best.get('test_accuracy', 0):.2f}% accuracy

2. Label efficiency: """
    
    if 0.1 in by_budget and 1.0 in by_budget:
        acc_10 = np.mean(by_budget[0.1])
        acc_100 = np.mean(by_budget[1.0])
        summary += f"With only 10% labels, models achieved {acc_10:.1f}% "
        summary += f"({acc_10/acc_100*100:.1f}% of full-label performance)\n"
    
    summary += f"""
3. Training efficiency: Experiments completed successfully
   Average training time per model: {np.mean([r.get('time', 0) for r in successful])/60:.1f} minutes

 OUTPUT FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Results table (LaTeX):  outputs/plots/thesis_results_table.tex
- Detailed JSON:          outputs/comparison_results/

╚══════════════════════════════════════════════════════════════════╝
"""
    
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"\n📄 Executive summary saved: {output_file}")
    
    return output_file

def main():
    """Generate all thesis materials"""
    print("="*70)
    print(" THESIS ANALYSIS & VISUALIZATION SUITE")
    print("="*70)
    
    # Load results
    results = load_comparison_results()
    
    # 1. Create summary table
    print("\n Creating summary table...")
    df = create_summary_table(results)
    print(df.to_string(index=False))
    
    # 2. Save LaTeX table
    print("\n Generating LaTeX table...")
    latex_path = save_latex_table(df)
    
    # 3. Generate plots
    print("\n Generating label efficiency curve...")
    plot_label_efficiency_curve(results)
    
    print("\n Generating model comparison chart...")
    plot_model_comparison_bar(results)
    
    # 4. Executive summary
    print("\n Generating executive summary...")
    generate_executive_summary(results)
    # Append AULC summary (<=10% by default) to executive summary
    append_efficiency_summary_to_executive(results)
    
    print("\n" + "="*70)
    print(" THESIS MATERIALS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\n All outputs saved in: outputs/plots/")
    print("   - Use PNG files for presentations")
    print("\n Ready for your thesis chapters!")

if __name__ == "__main__":
    main()
