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
    'description': 'Safe for L40S, ~7-8 hours for full dataset',
    # Learning rate configurations for different training regimes
    'learning_rates': {
        'linear_probe': {
            'base_lr': 0.0008,
            'lr_multiplier': 1.0,  # No additional scaling needed
            'warmup_epochs': 2,
            'warmup_start_lr': 1e-5
        },
        'fine_tune': {
            'base_lr': 0.0008,
            'lr_multiplier': 1.0,
            'warmup_epochs': 2,
            'warmup_start_lr': 1e-5
        },
        'supervised': {
            'base_lr': 0.0005,
            'lr_multiplier': 1.0,
            'warmup_epochs': 3,
            'warmup_start_lr': 1e-6
        }
    }
}

L40S_AGGRESSIVE = {
    'name': 'L40S 24GB Aggressive',
    'per_step_batch': 192,
    'gradient_accum_steps': 2,
    'effective_batch': 384,
    'num_workers': 8,
    'description': 'WARNING: May OOM on large models, test first. ~5-6 hours if successful',
    'learning_rates': {
        'linear_probe': {
            'base_lr': 0.0008,
            'lr_multiplier': 1.2,  # Slightly higher for larger batch
            'warmup_epochs': 2,
            'warmup_start_lr': 1e-5
        },
        'fine_tune': {
            'base_lr': 0.0008,
            'lr_multiplier': 1.2,
            'warmup_epochs': 2,
            'warmup_start_lr': 1e-5
        },
        'supervised': {
            'base_lr': 0.0005,
            'lr_multiplier': 1.2,
            'warmup_epochs': 3,
            'warmup_start_lr': 1e-6
        }
    }
}

A100_OPTIMAL = {
    'name': 'A100 80GB Optimal',
    'per_step_batch': 256,
    'gradient_accum_steps': 1,  # No accumulation needed!
    'effective_batch': 256,
    'num_workers': 18,
    'description': 'Perfect for A100, ~3-4 hours for full dataset',
    'learning_rates': {
        'linear_probe': {
            'base_lr': 0.0008,
            'lr_multiplier': 1.5,  # Higher LR for A100's capabilities
            'warmup_epochs': 2,
            'warmup_start_lr': 1e-5
        },
        'fine_tune': {
            'base_lr': 0.0008,
            'lr_multiplier': 1.5,
            'warmup_epochs': 2,
            'warmup_start_lr': 1e-5
        },
        'supervised': {
            'base_lr': 0.0005,
            'lr_multiplier': 1.5,
            'warmup_epochs': 3,
            'warmup_start_lr': 1e-6
        }
    }
}

A100_ULTRA = {
    'name': 'A100 80GB Ultra-Fast',
    'per_step_batch': 384,
    'gradient_accum_steps': 1,
    'effective_batch': 384,
    'num_workers': 18,
    'description': 'Fastest possible, ~2-3 hours. Requires LR adjustment (×1.5)',
    'auto_scale_lr': True,
    'lr_multiplier': 1.5,
    'learning_rates': {
        'linear_probe': {
            'base_lr': 0.0008,
            'lr_multiplier': 2.0,  # Higher multiplier for ultra-fast training
            'warmup_epochs': 2,
            'warmup_start_lr': 1e-5
        },
        'fine_tune': {
            'base_lr': 0.0008,
            'lr_multiplier': 2.0,
            'warmup_epochs': 2,
            'warmup_start_lr': 1e-5
        },
        'supervised': {
            'base_lr': 0.0005,
            'lr_multiplier': 2.0,
            'warmup_epochs': 3,
            'warmup_start_lr': 1e-6
        }
    }
}

# H100 presets
H100_OPTIMAL = {
    'name': 'H100 80GB Optimal',
    'per_step_batch': 384,
    'gradient_accum_steps': 1,
    'effective_batch': 384,
    'num_workers': 18,
    'description': 'Great for H100, ~2-3 hours for full dataset',
    'learning_rates': {
        'linear_probe': {
            'base_lr': 0.0008,
            'lr_multiplier': 1.8,  # H100 can handle higher LRs
            'warmup_epochs': 2,
            'warmup_start_lr': 1e-5
        },
        'fine_tune': {
            'base_lr': 0.0008,
            'lr_multiplier': 1.8,
            'warmup_epochs': 2,
            'warmup_start_lr': 1e-5
        },
        'supervised': {
            'base_lr': 0.0005,
            'lr_multiplier': 1.8,
            'warmup_epochs': 3,
            'warmup_start_lr': 1e-6
        }
    }
}

H100_ULTRA = {
    'name': 'H100 80GB Ultra-Fast',
    'per_step_batch': 512,
    'gradient_accum_steps': 1,
    'effective_batch': 512,
    'num_workers': 18,
    'description': 'Fastest possible on H100, ~1.5-2.5 hours. Requires LR adjustment (×2.0). WARNING: May OOM on very large models, test first',
    'auto_scale_lr': True,
    'lr_multiplier': 2.0,
    'learning_rates': {
        'linear_probe': {
            'base_lr': 0.0008,
            'lr_multiplier': 2.5,  # Maximum multiplier for H100 ultra
            'warmup_epochs': 2,
            'warmup_start_lr': 1e-5
        },
        'fine_tune': {
            'base_lr': 0.0008,
            'lr_multiplier': 2.5,
            'warmup_epochs': 2,
            'warmup_start_lr': 1e-5
        },
        'supervised': {
            'base_lr': 0.0005,
            'lr_multiplier': 2.5,
            'warmup_epochs': 3,
            'warmup_start_lr': 1e-6
        }
    }
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

print(f"Using: {GPU_CONFIG['name']}")
print(f" {GPU_CONFIG['description']}")
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

def apply_gpu_learning_rate_config(active_config, gpu_config, training_regime, model_type='cnn'):
    """
    Apply GPU-specific learning rate configuration to active_config
    
    Args:
        active_config: The configuration object to modify
        gpu_config: GPU configuration dictionary (e.g., A100_OPTIMAL)
        training_regime: 'linear_probe', 'fine_tune', or 'supervised'
        model_type: 'cnn', 'dinov2', or 'vit'
    
    Returns:
        Modified active_config with GPU-optimized learning rates
    """
    if 'learning_rates' not in gpu_config:
        print(f"WARNING: No learning_rates found in GPU config: {gpu_config.get('name', 'Unknown')}")
        return active_config
    
    lr_config = gpu_config['learning_rates'].get(training_regime)
    if not lr_config:
        print(f"WARNING: No learning rate config for regime '{training_regime}' in GPU config")
        return active_config
    
    # Apply base learning rate and multiplier
    base_lr = lr_config['base_lr']
    lr_multiplier = lr_config['lr_multiplier']
    final_lr = base_lr * lr_multiplier
    
    # Model-specific adjustments
    if model_type == 'vit' and training_regime == 'supervised':
        # ViT supervised training needs slightly lower LR for stability
        final_lr *= 0.8
    
    active_config.LEARNING_RATE = float(final_lr)
    
    # Apply warmup configuration
    active_config.USE_WARMUP = True
    active_config.WARMUP_EPOCHS = lr_config['warmup_epochs']
    active_config.WARMUP_START_LR = lr_config['warmup_start_lr']
    
    print(f"GPU LR Config Applied:")
    print(f"   Base LR: {base_lr:.6f}")
    print(f"   Multiplier: {lr_multiplier:.2f}")
    print(f"   Final LR: {final_lr:.6f}")
    print(f"   Warmup: {lr_config['warmup_epochs']} epochs, start={lr_config['warmup_start_lr']:.1e}")
    
    return active_config

# =============================================================================
# EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("GPU CONFIGURATION PRESETS")
    print("="*80)
    
    configs = [
        L40S_CONSERVATIVE,
        L40S_AGGRESSIVE,
        A100_OPTIMAL,
        A100_ULTRA,
        H100_OPTIMAL,
        H100_ULTRA,
    ]
    
    for cfg in configs:
        print(f"\n{cfg['name']}")
        print(f"  Per-step batch: {cfg['per_step_batch']}")
        print(f"  Gradient accum: {cfg['gradient_accum_steps']} steps")
        print(f"  Effective batch: {cfg['effective_batch']}")
        print(f"  Num workers: {cfg['num_workers']}")
        print(f"   {cfg['description']}")
    
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
