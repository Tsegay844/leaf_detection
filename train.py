from ultralytics import YOLO
from nn.esp_tasks import custom_parse_model
import ultralytics.nn.tasks as tasks

def Train(pretrained_path=None, dataset="datasets/grape_leaf/data.yaml", imgsz=320, **kwargs):
    """
    Train espdet_pico on customized dataset optimized for grape leaf disease detection.
    :param pretrained_path: the path of pretrained .pt file, default is None.
    :param imgsz: input image size (320 recommended for leaf detection)
    :return:
    """
    tasks.parse_model = custom_parse_model  # add ESP-customized block
    # load the model
    if pretrained_path not in [None, 'None']: # use pretrained weights
        model = YOLO(pretrained_path)
    else:
        model = YOLO('cfg/models/espdet_pico.yaml') # # build a new model from YAML if you don't need to load a pretrained model
    
    # Professional configuration for LEAF LOCALIZATION in two-stage pipeline:
    # Stage 1: Leaf localization (espdet_pico) → Stage 2: Disease classification (MobileNetV2)
    # Optimized for HIGH PRECISION & BALANCED RECALL for ESP32-S3 + LoRaWAN deployment
    train_setting = dict(
        # Dataset and basic training parameters
        data=dataset,
        epochs=2000,  # Moderate training - localization is simpler than disease detection
        patience=200,  # Good patience for convergence
        imgsz=imgsz,  # 320x320 for leaf boundary clarity
        batch=64,  # Larger batch OK for localization (simpler task than disease detection)
        device='0',
        workers=8,
        seed=42,  # Reproducibility
        
        # Learning rate optimization
        lr0=0.001,  # Moderate LR - localization is easier than fine-grained disease detection
        lrf=0.01,   # Standard final LR multiplier (0.00001)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,  # Standard warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Optimizer - AdamW for consistent convergence
        optimizer='AdamW',
        
        # Color augmentation - REDUCED (leaf detection less color-sensitive than disease)
        # Focus: Detect leaf shape/structure, not disease color variations
        hsv_h=0.015,  # Minimal hue shift - green leaves stay relatively consistent
        hsv_s=0.4,    # Moderate saturation - maintain leaf appearance
        hsv_v=0.4,    # Brightness variation for lighting conditions
        
        # Geometric augmentation - vineyard camera scenarios
        degrees=15.0,     # Moderate rotation - camera angles in vineyard rows
        translate=0.1,    # Standard translation
        scale=0.5,        # Scale variation (0.5-1.5x) - near/far leaves
        shear=2.0,        # Light shear - perspective when camera not perpendicular
        perspective=0.0001,  # Minimal perspective
        flipud=0.5,       # Vertical flip - hanging leaves, wind movement
        fliplr=0.5,       # Horizontal flip
        
        # Advanced augmentation - BALANCED for precision
        mosaic=1.0,        # Full mosaic for multi-scale robustness
        mixup=0.05,        # REDUCED - too much mixing hurts localization precision
        copy_paste=0.15,    # MODERATE - some augmentation but avoid artifacts
        close_mosaic=80,   # Disable augmentation for clean fine-tuning
        
        # Training strategy - PRECISION FOCUS
        rect=False,        # Square images for consistency
        cos_lr=True,       # Cosine annealing for smooth convergence
        label_smoothing=0.0,  # NO smoothing - want confident predictions for sampling
        
        # Loss weights - PRECISION-OPTIMIZED for leaf localization
        box=7.5,   # Standard box loss - leaf boundaries are clearer than disease spots
        cls=0.6,   # INCREASED class loss - want confident "this is a leaf" predictions
        dfl=1.5,   # Standard DFL weight
        
        # Confidence threshold tuning - IMPORTANT for precision
        # During validation, use higher thresholds to filter low-confidence detections
        # This simulates the "top confidence" sampling strategy
        
        # Validation and checkpointing
        val=True,
        save=True,
        save_period=100,  # Save checkpoints every 100 epochs
        plots=True,
        exist_ok=True,
        
        # Performance optimization
        amp=True,        # Mixed precision for speed
        fraction=1.0,    # Use full dataset
        
        # Monitoring and debugging
        verbose=True,
        deterministic=False,  # Allow non-deterministic ops for speed
    )
    train_setting.update(kwargs)
    results = model.train(**train_setting)
       # Training complete
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best weights:  runs/detect/grape_leaf_localization/weights/best.pt")
    print(f"Last weights:  runs/detect/grape_leaf_localization/weights/last.pt")
    print(f"\nNext steps:")
    print(f"  1. Run val.py to validate with ESP32 deployment parameters")
    print(f"  2. Export to TFLite: model.export(format='tflite', imgsz=320)")
    print(f"  3. Quantize for ESP32: Use TFLite INT8 quantization")
    print("="*70 + "\n")
    

    return results

if __name__ == '__main__':
    Train()

