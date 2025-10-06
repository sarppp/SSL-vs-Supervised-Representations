#!/usr/bin/env python3
"""
📋 Unified Log Creator - Consolidate Multiple Log Files per Experiment
====================================================================
This script combines the training, model_setup, and evaluation logs 
into single unified logs per experiment for easier analysis.
"""
import os
import glob
from pathlib import Path
from datetime import datetime

def consolidate_logs():
    """Combine multiple log files per experiment into unified logs"""
    
    logs_dir = Path("outputs/logs")
    unified_dir = logs_dir / "unified"
    unified_dir.mkdir(exist_ok=True)
    
    # Find all unique model experiments
    log_files = list(logs_dir.glob("*.log"))
    
    # Group by model name
    experiments = {}
    
    for log_file in log_files:
        filename = log_file.name
        
        # Skip already processed files
        if filename.startswith('unified_') or filename == 'comparison.log':
            continue
            
        # Extract model identifier
        if filename.startswith('training_'):
            model_id = filename.replace('training_', '').replace('.log', '')
        elif filename.startswith('model_setup_'):
            model_id = filename.replace('model_setup_', '').replace('.log', '')
        elif filename.startswith('evaluation_'):
            model_id = filename.replace('evaluation_', '').replace('.log', '')
        else:
            continue
            
        if model_id not in experiments:
            experiments[model_id] = {}
            
        # Categorize the log file
        if filename.startswith('training_'):
            experiments[model_id]['training'] = log_file
        elif filename.startswith('model_setup_'):
            experiments[model_id]['setup'] = log_file
        elif filename.startswith('evaluation_'):
            experiments[model_id]['evaluation'] = log_file
    
    print("📋 CONSOLIDATING LOGS")
    print("=" * 50)
    
    # Create unified logs
    for model_id, log_types in experiments.items():
        unified_file = unified_dir / f"unified_{model_id}.log"
        
        print(f" Creating: {unified_file.name}")
        
        with open(unified_file, 'w') as outfile:
            # Header
            outfile.write("=" * 80 + "\n")
            outfile.write(f"🔬 UNIFIED EXPERIMENT LOG: {model_id}\n")
            outfile.write(f"📅 Consolidated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            outfile.write("=" * 80 + "\n\n")
            
            # Model Setup Section
            if 'setup' in log_types:
                outfile.write(" MODEL SETUP LOGS\n")
                outfile.write("-" * 40 + "\n")
                try:
                    with open(log_types['setup'], 'r') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f" Error reading setup log: {e}\n")
                outfile.write("\n" + "=" * 80 + "\n\n")
            
            # Training Section  
            if 'training' in log_types:
                outfile.write("TRAINING LOGS\n")
                outfile.write("-" * 40 + "\n")
                try:
                    with open(log_types['training'], 'r') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f" Error reading training log: {e}\n")
                outfile.write("\n" + "=" * 80 + "\n\n")
            
            # Evaluation Section
            if 'evaluation' in log_types:
                outfile.write("EVALUATION LOGS\n") 
                outfile.write("-" * 40 + "\n")
                try:
                    with open(log_types['evaluation'], 'r') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f" Error reading evaluation log: {e}\n")
                outfile.write("\n" + "=" * 80 + "\n\n")
            
            # Summary
            outfile.write("📋 EXPERIMENT SUMMARY\n")
            outfile.write("-" * 40 + "\n")
            outfile.write(f"Model ID: {model_id}\n")
            outfile.write(f"Log sections: {', '.join(log_types.keys())}\n")
            outfile.write(f"Unified log: {unified_file}\n")
            
        print(f"   Combined {len(log_types)} log files")
    
    print(f"\n📁 Unified logs saved to: {unified_dir}")
    print(f"Now you only need to check one file per experiment!")
    
    # Create index file
    index_file = unified_dir / "INDEX.md"
    with open(index_file, 'w') as f:
        f.write("# 📋 Unified Logs Index\n\n")
        f.write("## Available Experiments\n\n")
        
        for model_id in sorted(experiments.keys()):
            f.write(f"- `unified_{model_id}.log` - Complete log for {model_id}\n")
        
        f.write(f"\n## How to Use\n\n")
        f.write("Each unified log contains:\n")
        f.write("1. **Model Setup** - Model creation and configuration\n")
        f.write("2. **Training** - Training progress, epochs, losses\n") 
        f.write("3. **Evaluation** - Test results and final metrics\n\n")
        f.write("Instead of opening 3 separate files, everything is in one place!\n")
    
    print(f"📖 Index created: {index_file}")

if __name__ == "__main__":
    consolidate_logs()
