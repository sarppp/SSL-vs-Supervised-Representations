import torch
import torch.nn as nn
import torchvision.models as models
import os
import types
from ..utils.logger_manager import ModelSetupLogger
from ..config import config_paths

def get_model_identifier(model_name=None, config=None):
    """Generate a unique model identifier to prevent conflicts between model types.
    
    The identifier includes:
    - Model family and variant
    - Training strategy (fine_tune/freeze/full_train)
    - Label percentage (few-shot learning)
    
    Examples:
    - dinov2_vitb14_fine_tune_labels10pct
    - efficientnet_b4_freeze_labels100pct
    - vit_base_patch16_224_full_train_labels50pct
    """
    if config is None:
        raise ValueError("You must provide a config module!")
    model_name = model_name or config.MODEL_NAME
    
    # Create model family identifier to avoid conflicts
    if model_name.startswith('dinov2'):
        variant = model_name.replace('dinov2_', '')
        base_id = f"dinov2_{variant}"
    elif model_name.startswith('efficientnet'):
        variant = model_name.replace('efficientnet_', '')
        base_id = f"efficientnet_{variant}"
    elif model_name.startswith('vit_'):
        # Keep full vit name for clarity
        base_id = model_name
    elif model_name.startswith('resnet'):
        base_id = model_name
    else:
        base_id = model_name

    # Determine training strategy (honor explicit TRAINING_REGIME when provided)
    explicit_regime = getattr(config, 'TRAINING_REGIME', None)
    freeze_backbone = getattr(config, 'FREEZE_BACKBONE', False)
    unfreeze_after_epoch = getattr(config, 'UNFREEZE_AFTER_EPOCH', None)
    
    if isinstance(explicit_regime, str) and explicit_regime.strip():
        # Normalize common names
        regime = explicit_regime.lower()
        if regime in ["linear_probe", "linearprobe", "probe", "frozen"]:
            strategy = "linear_probe"
        elif regime in ["fine_tune", "finetune", "fine"]:
            strategy = "fine_tune"
        elif regime in ["supervised", "full", "full_train", "end_to_end"]:
            strategy = "full_train"
        else:
            # Fallback to heuristic below if unrecognized
            strategy = None
    else:
        strategy = None
    
    if strategy is None:
        if freeze_backbone and unfreeze_after_epoch is not None and unfreeze_after_epoch > 0:
            strategy = "fine_tune"  # Starts frozen, then unfreezes
        elif freeze_backbone:
            strategy = "freeze"  # Always frozen
        else:
            strategy = "full_train"  # Never frozen
    
    # Add few-shot label for data subset identification
    few_shot_mode = getattr(config, 'FEW_SHOT_MODE', None)
    if few_shot_mode == 'percentage':
        few_shot_value = getattr(config, 'FEW_SHOT_VALUE', 1.0)
        percentage = int(few_shot_value * 100)
    else:
        percentage = 100  # Default when few-shot is not used
    
    # Build final identifier
    return f"{base_id}_{strategy}_labels{percentage}pct"

def create_improved_classifier(in_features, num_classes, model_type="cnn", config=None):
    """Create an improved classification head with modern components."""
    dropout = getattr(config, 'DROPOUT', 0.2)
    use_simple_head = getattr(config, 'USE_SIMPLE_HEAD', True)
    
    if use_simple_head:
        # Simple head for comparison/ablation studies
        if model_type == "dinov2":
            return nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
    
    # Advanced head (default)
    if model_type == "dinov2":
        # DINOv2: LayerNorm is crucial for Vision Transformers
        return nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),  # Intermediate layer
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),  # Reduced dropout for second layer
            nn.Linear(in_features // 2, num_classes)
        )
    else:
        # CNN: BatchNorm works better than LayerNorm for CNNs
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),  # Intermediate layer
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),  # Reduced dropout for second layer  
            nn.Linear(in_features // 2, num_classes)
        )

def create_model(num_classes, model_name=None, config=None, setup_logger=None):
    """Create a pre-trained model for classification."""
    if config is None:
        raise ValueError("You must provide a config module!")
    model_name = model_name or config.MODEL_NAME
    
    # Generate full model identifier
    model_id = get_model_identifier(model_name, config)
    
    # Create logger if not provided
    if setup_logger is None:
        setup_logger = ModelSetupLogger(model_id, model_name=model_name)
    
    setup_logger.log_model_creation_start(model_name, model_id)
    
    if model_name in ['efficientnet_b0', 'efficientnet_b3', 'efficientnet_b4']:
        # Support EfficientNet B0, B3, and B4
        # Determine freeze setting from config (default False for backward compat)
        freeze_backbone = getattr(config, 'FREEZE_BACKBONE', False)

        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='DEFAULT')
        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(weights='DEFAULT')
        elif model_name == 'efficientnet_b4':
            model = models.efficientnet_b4(weights='DEFAULT')
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {model_name}")
            
        # Freeze backbone if requested (all feature layers)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        # Attach unfreeze_backbone method so training loop can handle progressive unfreezing
        def _unfreeze_backbone(self):
            for param in self.features.parameters():
                param.requires_grad = True
            print("🔥 CNN backbone unfrozen for fine-tuning")

        # Dynamically attach method (use setattr to avoid Pyright argument-type warning)
        setattr(model, 'unfreeze_backbone', types.MethodType(_unfreeze_backbone, model))

        # Robustly get in_features
        if isinstance(model.classifier, nn.Sequential):
            for layer in model.classifier:
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
            else:
                raise ValueError("No Linear layer found in EfficientNet classifier!")
        else:
            in_features = model.classifier.in_features

        # Use improved classifier
        model.classifier = create_improved_classifier(in_features, num_classes, "cnn", config)

        # CNNs generally do not support gradient checkpointing; log explicitly if requested
        use_checkpointing = getattr(config, 'USE_GRADIENT_CHECKPOINTING', False)
        if use_checkpointing:
            setup_logger.log_gradient_checkpointing(False, False)
    
    elif model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')

        freeze_backbone = getattr(config, 'FREEZE_BACKBONE', False)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # Always keep layer4 trainable for resnet if frozen earlier
        #for param in model.layer4.parameters():
            #param.requires_grad = True
        # NOTE: We intentionally keep the entire ResNet backbone frozen (including layer4)
        # until the scheduled unfreeze epoch.  This mirrors the DINOv2 strategy so that
        # both model families undergo the same progressive-fine-tuning regime.

        # Attach unfreeze_backbone for ResNet
        def _unfreeze_backbone(self):
            for param in self.parameters():
                param.requires_grad = True
            print("🔥 CNN backbone (ResNet) unfrozen for fine-tuning")

        # Dynamically attach method (use setattr to avoid Pyright argument-type warning)
        setattr(model, 'unfreeze_backbone', types.MethodType(_unfreeze_backbone, model))
        
        # Use improved classifier as a separate attribute to avoid type errors
        in_features = model.fc.in_features
        model.improved_classifier = create_improved_classifier(in_features, num_classes, "cnn", config)
        # Optionally, keep the original fc for compatibility, but forward should use improved_classifier
        # You may need to override the forward method elsewhere to use model.improved_classifier(x)
    
    elif model_name.startswith('vit_'):
        # ViT models support
        try:
            import timm
        except ImportError:
            raise ImportError("timm library is required for ViT models. Install with: pip install timm")
        
        # Use model-specific config if available
        model_configs = getattr(config, 'MODEL_CONFIGS', {})
        if model_name in model_configs:
            model_config = model_configs[model_name]
            setup_logger.log_model_config(model_name, model_config)
            # Override config values with model-specific ones
            config.LEARNING_RATE = model_config['learning_rate']
            config.WEIGHT_DECAY = model_config['weight_decay']
            config.DROPOUT = model_config['dropout']
        
        try:
            # Load pre-trained ViT model from timm
            vit_model = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 removes head
            vit_model.eval()
            
            # Get the embedding dimension
            if hasattr(vit_model, 'embed_dim'):
                embed_dim = vit_model.embed_dim
            elif hasattr(vit_model, 'num_features'):
                embed_dim = vit_model.num_features
            else:
                # Default dimensions for common ViT variants
                if 'tiny' in model_name:
                    embed_dim = 192
                elif 'small' in model_name:
                    embed_dim = 384
                elif 'base' in model_name:
                    embed_dim = 768
                elif 'large' in model_name:
                    embed_dim = 1024
                else:
                    raise ValueError(f"Unknown ViT variant: {model_name}. Expected one of: tiny, small, base, large")
            
            # Create classifier head
            classifier = create_improved_classifier(embed_dim, num_classes, "vit", config)
            
            # Create wrapper class similar to DINOv2
            class ViTClassifier(nn.Module):
                def __init__(self, backbone, classifier, embed_dim):
                    super().__init__()
                    self.backbone = backbone
                    self.classifier = classifier
                    self.embed_dim = embed_dim
                    
                    # Freeze backbone initially if specified
                    freeze_backbone = getattr(config, 'FREEZE_BACKBONE', False)
                    if freeze_backbone:
                        for param in self.backbone.parameters():
                            param.requires_grad = False
                        setup_logger.log_backbone_frozen()
                
                def forward(self, x):
                    # Get features from ViT backbone
                    features = self.backbone(x)
                    return self.classifier(features)
                
                def unfreeze_backbone(self):
                    """Unfreeze backbone for fine-tuning"""
                    for param in self.backbone.parameters():
                        param.requires_grad = True
                    print("🔥 ViT backbone unfrozen for fine-tuning")
            
            model = ViTClassifier(vit_model, classifier, embed_dim)
            
            # Enable gradient checkpointing if specified
            use_checkpointing = getattr(config, 'USE_GRADIENT_CHECKPOINTING', False)
            if use_checkpointing:
                try:
                    # Try different methods for enabling gradient checkpointing
                    checkpointing_enabled = False
                    
                    # Method 1: set_grad_checkpointing (newer versions)
                    if hasattr(vit_model, 'set_grad_checkpointing'):
                        vit_model.set_grad_checkpointing(True)
                        checkpointing_enabled = True
                    
                    # Method 2: gradient_checkpointing_enable (alternative method)
                    elif hasattr(vit_model, 'gradient_checkpointing_enable'):
                        vit_model.gradient_checkpointing_enable()
                        checkpointing_enabled = True
                    
                    # Method 3: Check if it's a HuggingFace-style model
                    elif hasattr(vit_model, 'config') and hasattr(vit_model.config, 'use_cache'):
                        vit_model.config.use_cache = False
                        checkpointing_enabled = True
                    
                    # Method 4: Try to enable on the transformer blocks
                    elif hasattr(vit_model, 'blocks'):
                        for block in vit_model.blocks:
                            if hasattr(block, 'gradient_checkpointing'):
                                block.gradient_checkpointing = True
                                checkpointing_enabled = True
                    
                    # Method 5: Try to enable on the encoder blocks (ViT specific)
                    elif hasattr(vit_model, 'encoder') and hasattr(vit_model.encoder, 'layers'):
                        for layer in vit_model.encoder.layers:
                            if hasattr(layer, 'gradient_checkpointing'):
                                layer.gradient_checkpointing = True
                                checkpointing_enabled = True
                    
                    if checkpointing_enabled:
                        setup_logger.log_gradient_checkpointing(True, True)
                    else:
                        setup_logger.log_gradient_checkpointing(True, False, "No gradient checkpointing method found")
                        
                except Exception as e:
                    setup_logger.log_gradient_checkpointing(True, False, str(e))
            
        except Exception as e:
            error_msg = f"Error loading ViT model: {e}\nMake sure you have timm installed: pip install timm"
            setup_logger.log_error(error_msg)
            raise
    
    elif model_name.startswith('dinov2'):
        # Use model-specific config if available
        model_configs = getattr(config, 'MODEL_CONFIGS', {})
        if model_name in model_configs:
            model_config = model_configs[model_name]
            setup_logger.log_model_config(model_name, model_config)
            # Override config values with model-specific ones
            config.LEARNING_RATE = model_config['learning_rate']
            config.WEIGHT_DECAY = model_config['weight_decay']
            config.DROPOUT = model_config['dropout']
        
        try:
            # Load pre-trained DINOv2 model
            dinov2_model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)  # type: ignore
            # Ensure it's in eval mode initially
            dinov2_model.eval()  # type: ignore
            
            # Get the embedding dimension
            if 'vits14' in model_name:
                embed_dim = 384
            elif 'vitb14' in model_name:
                embed_dim = 768
            elif 'vitl14' in model_name:
                embed_dim = 1024
            elif 'vitg14' in model_name:
                embed_dim = 1536
            else:
                raise ValueError(f"Unknown DINOv2 variant: {model_name}")
            
            # Freeze backbone if specified in config
            freeze_backbone = getattr(config, 'FREEZE_BACKBONE', True)
            setup_logger.log_dinov2_info(model_name, embed_dim, freeze_backbone)
            
            if freeze_backbone:
                for param in dinov2_model.parameters():  # type: ignore
                    param.requires_grad = False
            
            # Create improved classification head for DINOv2
            classifier = create_improved_classifier(embed_dim, num_classes, "dinov2", config)
            
            # Create full model class
            class DINOv2Classifier(nn.Module):
                def __init__(self, backbone, classifier, embed_dim):
                    super().__init__()
                    self.backbone = backbone
                    self.classifier = classifier
                    self.embed_dim = embed_dim
                    
                def forward(self, x):
                    # Get [CLS] token embeddings
                    features = self.backbone(x)  # Shape: [batch_size, embed_dim]
                    return self.classifier(features)
                
                def unfreeze_backbone(self):
                    """Unfreeze backbone for fine-tuning"""
                    for param in self.backbone.parameters():
                        param.requires_grad = True
                    print("🔥 Backbone unfrozen for fine-tuning")
            
            model = DINOv2Classifier(dinov2_model, classifier, embed_dim)
            
            # Enable gradient checkpointing if specified
            use_checkpointing = getattr(config, 'USE_GRADIENT_CHECKPOINTING', False)
            if use_checkpointing:
                try:
                    # Try different methods for enabling gradient checkpointing
                    checkpointing_enabled = False
                    
                    # Method 1: set_grad_checkpointing (newer versions)
                    if hasattr(dinov2_model, 'set_grad_checkpointing'):
                        dinov2_model.set_grad_checkpointing(True)  # type: ignore
                        checkpointing_enabled = True
                    
                    # Method 2: gradient_checkpointing_enable (alternative method)
                    elif hasattr(dinov2_model, 'gradient_checkpointing_enable'):
                        dinov2_model.gradient_checkpointing_enable()  # type: ignore
                        checkpointing_enabled = True
                    
                    # Method 3: Check if it's a HuggingFace-style model
                    elif hasattr(dinov2_model, 'config') and hasattr(dinov2_model.config, 'use_cache'):
                        dinov2_model.config.use_cache = False  # type: ignore
                        checkpointing_enabled = True
                    
                    # Method 4: Try to enable on the transformer blocks
                    elif hasattr(dinov2_model, 'blocks'):
                        for block in dinov2_model.blocks:
                            if hasattr(block, 'gradient_checkpointing'):
                                block.gradient_checkpointing = True
                                checkpointing_enabled = True
                    
                    if checkpointing_enabled:
                        setup_logger.log_gradient_checkpointing(True, True)
                    else:
                        setup_logger.log_gradient_checkpointing(True, False, "No gradient checkpointing method found")
                        
                except Exception as e:
                    setup_logger.log_gradient_checkpointing(True, False, str(e))
            
        except Exception as e:
            error_msg = f"Error loading DINOv2 model: {e}\nMake sure you have internet connection for torch.hub download"
            setup_logger.log_error(error_msg)
            raise
    
    else:
        raise ValueError(f"Model {model_name} not supported. Choose from {config.AVAILABLE_MODELS}")
    
    # Model compilation for PyTorch 2.0+ speedup (optional)
    compile_model = getattr(config, 'COMPILE_MODEL', False)
    if compile_model:
        try:
            # Check if torch.compile is available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                model = torch.compile(model)  # type: ignore
                setup_logger.log_model_compilation(True)
            else:
                setup_logger.log_model_compilation(False)
        except Exception as e:
            setup_logger.log_model_compilation(False, str(e))
    
    setup_logger.log_model_creation_complete(model_name, num_classes)
    return model

def create_optimizer(model_parameters, config=None, setup_logger=None):
    """Create optimizer based on config."""
    if config is None:
        raise ValueError("You must provide a config module!")
    optimizer_name = config.OPTIMIZER.lower()
    params = config.OPTIMIZER_PARAMS[optimizer_name].copy()
    
    if setup_logger:
        setup_logger.log_optimizer_creation(optimizer_name, params)
    
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(model_parameters, **params)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(model_parameters, **params)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model_parameters, **params)
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(model_parameters, **params)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported. Choose from: {list(config.OPTIMIZER_PARAMS.keys())}")

def create_scheduler(optimizer, config=None, setup_logger=None):
    """Create scheduler based on config with optional warmup support."""
    if config is None:
        raise ValueError("You must provide a config module!")
    scheduler_name = config.SCHEDULER.lower()
    params = config.SCHEDULER_PARAMS[scheduler_name].copy()
    
    if setup_logger:
        setup_logger.log_scheduler_creation(scheduler_name, params)
    
    # Create main scheduler
    main_scheduler = None
    if scheduler_name == 'plateau':
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    elif scheduler_name == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    elif scheduler_name == 'step':
        main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **params)
    elif scheduler_name == 'linear':
        main_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **params)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported. Choose from: {list(config.SCHEDULER_PARAMS.keys())}")
    
    # Check if warmup is enabled in config (won't break if not defined)
    use_warmup = getattr(config, 'USE_WARMUP', False)
    
    if use_warmup:
        warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 3)
        warmup_start_lr = getattr(config, 'WARMUP_START_LR', 1e-6)
        main_lr = getattr(config, 'LEARNING_RATE', optimizer.param_groups[0]['lr'])
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=warmup_start_lr / main_lr,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        if setup_logger:
            warmup_info = {
                'epochs': warmup_epochs,
                'start_lr': warmup_start_lr,
                'main_lr': main_lr
            }
            setup_logger.log_scheduler_creation(scheduler_name, params, True, warmup_info)
        
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    
    return main_scheduler

def setup_training(model, class_weights_tensor, config=None, setup_logger=None):
    """Setup model, criterion, optimizer, and scheduler."""
    
    if config is None:
        raise ValueError("You must provide a config module!")
    
    # Create logger if not provided
    if setup_logger is None:
        model_name = getattr(config, 'MODEL_NAME', 'unknown')
        model_id = get_model_identifier(model_name, config)
        setup_logger = ModelSetupLogger(model_id, model_name=model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    class_weights_tensor = class_weights_tensor.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Use config-based optimizer and scheduler
    optimizer = create_optimizer(model.parameters(), config, setup_logger)
    scheduler = create_scheduler(optimizer, config, setup_logger)
    
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Log setup completion
    warmup_enabled = getattr(config, 'USE_WARMUP', False)
    setup_logger.log_setup_complete(
        str(device), total_params, trainable_params, 
        config.OPTIMIZER, config.SCHEDULER, warmup_enabled
    )
    
    return model, criterion, optimizer, scheduler, device