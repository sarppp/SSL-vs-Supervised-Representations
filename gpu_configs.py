"""
GPU-Specific Configuration Presets
Quick way to switch between optimized settings for different GPUs
"""

# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

L40S_CONSERVATIVE = {
    'name': 'L40S 24GB Conservative',
    'per_step_batch': 128,
    'gradient_accum_steps': 2,
    'effective_batch': 256,
    'num_workers': 8,
    'description': '✅ Safe for L40S, ~7-8 hours for full dataset'
}

L40S_AGGRESSIVE = {
    'name': 'L40S 24GB Aggressive',
    'per_step_batch': 192,
    'gradient_accum_steps': 2,
    'effective_batch': 384,
    'num_workers': 8,
    'description': '⚠️ May OOM on large models, test first. ~5-6 hours if successful'
}

A100_OPTIMAL = {
    'name': 'A100 80GB Optimal',
    'per_step_batch': 256,
    'gradient_accum_steps': 1,  # No accumulation needed!
    'effective_batch': 256,
    'num_workers': 8,
    'description': '✅ Perfect for A100, ~3-4 hours for full dataset'
}

A100_ULTRA = {
    'name': 'A100 80GB Ultra-Fast',
    'per_step_batch': 384,
    'gradient_accum_steps': 1,
    'effective_batch': 384,
    'num_workers': 8,
    'description': '🚀 Fastest possible, ~2-3 hours. Requires LR adjustment (×1.5)'
}

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
"""
To use in compact_model_comparison.py:

from gpu_configs import A100_OPTIMAL as GPU_CONFIG

# Then in run_model():
PER_STEP_BATCH = GPU_CONFIG['per_step_batch']
GRADIENT_ACCUM_STEPS = GPU_CONFIG['gradient_accum_steps']
EFFECTIVE_BATCH = GPU_CONFIG['effective_batch']
NUM_WORKERS = GPU_CONFIG['num_workers']

print(f"🔧 Using: {GPU_CONFIG['name']}")
print(f"📝 {GPU_CONFIG['description']}")
"""

# =============================================================================
# LR SCALING HELPER
# =============================================================================

def get_scaled_lr(base_lr, base_batch=64, new_effective_batch=256, scaling_type='linear'):
    """
    Scale learning rate based on batch size
    
    Args:
        base_lr: Original learning rate (for batch 64)
        base_batch: Original batch size (default 64)
        new_effective_batch: New effective batch size
        scaling_type: 'linear' or 'sqrt'
    
    Returns:
        Scaled learning rate
    """
    ratio = new_effective_batch / base_batch
    
    if scaling_type == 'linear':
        # Linear scaling: new_lr = base_lr × (new_batch / old_batch)
        return base_lr * ratio
    elif scaling_type == 'sqrt':
        # Square root scaling: new_lr = base_lr × sqrt(new_batch / old_batch)
        import math
        return base_lr * math.sqrt(ratio)
    else:
        raise ValueError(f"Unknown scaling_type: {scaling_type}")

# =============================================================================
# EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("GPU CONFIGURATION PRESETS")
    print("="*80)
    
    configs = [L40S_CONSERVATIVE, L40S_AGGRESSIVE, A100_OPTIMAL, A100_ULTRA]
    
    for cfg in configs:
        print(f"\n{cfg['name']}")
        print(f"  Per-step batch: {cfg['per_step_batch']}")
        print(f"  Gradient accum: {cfg['gradient_accum_steps']} steps")
        print(f"  Effective batch: {cfg['effective_batch']}")
        print(f"  Num workers: {cfg['num_workers']}")
        print(f"  📝 {cfg['description']}")
    
    print("\n" + "="*80)
    print("LEARNING RATE SCALING EXAMPLES")
    print("="*80)
    
    base_lrs = {
        'Linear Probe': 0.001,
        'Fine-tune': 0.0005,
        'Supervised CNN': 0.001,
        'Supervised ViT': 0.0008,
    }
    
    print(f"\n{'Regime':<20} {'Base (64)':<12} {'Batch 256':<12} {'Batch 384':<12}")
    print("-"*60)
    
    for regime, base_lr in base_lrs.items():
        lr_256 = get_scaled_lr(base_lr, 64, 256, 'linear')
        lr_384 = get_scaled_lr(base_lr, 64, 384, 'linear')
        print(f"{regime:<20} {base_lr:<12.6f} {lr_256:<12.6f} {lr_384:<12.6f}")
    
    print("\n" + "="*80)
