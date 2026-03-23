#!/usr/bin/env python3
"""
Unified training script for OASIS-DC depth completion.
"""
import os
import sys
import argparse
import yaml
from types import SimpleNamespace
import subprocess
import random

import torch
import torch.optim as optim

from models.model import OASIS_DC
from config.schema import OasisCfg
from src.trainers.nyu_trainer import NYUTrainer
from src.trainers.kitti_trainer import KITTITrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_shot_adaptations(config: dict, shots: int) -> dict:
    """Apply shot-specific configuration adaptations"""
    if "shot_adaptations" in config and str(shots) in config["shot_adaptations"]:
        adaptations = config["shot_adaptations"][str(shots)]
        
        # Deep merge adaptations into config
        for section, settings in adaptations.items():
            if section not in config:
                config[section] = {}
            config[section].update(settings)
        
        print(f"Applied {shots}-shot adaptations")
    
    return config


def generate_train_list(dataset: str, data_root: str, shots: int, seed: int) -> str:
    """Generate training list filename based on shots and seed"""
    if dataset.lower() == "nyu":
        if shots == 1:
            return f"lists/nyu_train_1shot_seed{seed}.txt"
        elif shots == 10:
            return f"lists/nyu_train_10shot_seed{seed}.txt"
        elif shots == 100:
            return f"lists/nyu_train_100shot_seed{seed}.txt"
        else:
            # For custom shot counts, we'll need to generate the list
            return f"lists/nyu_train_{shots}shot_seed{seed}.txt"
    elif dataset.lower() == "kitti":
        if shots == 1:
            return f"lists/kitti_train_1shot_seed{seed}.txt"
        elif shots == 10:
            return f"lists/kitti_train_10shot_seed{seed}.txt"
        elif shots == 100:
            return f"lists/kitti_train_100shot_seed{seed}.txt"
        else:
            return f"lists/kitti_train_{shots}shot_seed{seed}.txt"
    
    return None


def create_nshot_list(dataset: str, data_root: str, shots: int, seed: int, output_path: str) -> bool:
    """Create n-shot training list by sampling from full dataset"""
    try:
        # First, create or find the full training list
        if dataset.lower() == "nyu":
            full_list_path = "lists/nyu_train_full.txt"
            if not os.path.exists(full_list_path):
                print(f"Creating full NYU training list: {full_list_path}")
                cmd = ["python", "make_list.py", "--dataset", "nyuv2", "--root", data_root, "--out", full_list_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error creating full list: {result.stderr}")
                    return False
        
        elif dataset.lower() == "kitti":
            full_list_path = "lists/kitti_train_full.txt"
            if not os.path.exists(full_list_path):
                print(f"Creating full KITTI training list: {full_list_path}")
                cmd = ["python", "make_list.py", "--dataset", "kitti", "--root", data_root, "--out", full_list_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error creating full list: {result.stderr}")
                    return False
        
        # Read full training list
        if not os.path.exists(full_list_path):
            print(f"Full training list not found: {full_list_path}")
            return False
            
        with open(full_list_path, 'r') as f:
            all_lines = [line.strip() for line in f if line.strip()]
        
        if len(all_lines) < shots:
            print(f"Warning: Requested {shots} shots but only {len(all_lines)} samples available")
            shots = len(all_lines)
        
        # Sample n-shot with seed
        random.seed(seed)
        selected_lines = random.sample(all_lines, shots)
        
        # Write n-shot list
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for line in selected_lines:
                f.write(line + '\n')
        
        print(f"Created {shots}-shot training list with seed {seed}: {output_path} ({len(selected_lines)} samples)")
        return True
        
    except Exception as e:
        print(f"Error creating n-shot list: {e}")
        return False


def ensure_train_list_exists(dataset: str, data_root: str, train_list_path: str, shots: int, seed: int) -> bool:
    """Ensure training list file exists, create if missing"""
    if os.path.exists(train_list_path):
        return True
    
    print(f"Training list not found: {train_list_path}")
    print(f"Attempting to create {shots}-shot list with seed {seed}...")
    
    return create_nshot_list(dataset, data_root, shots, seed, train_list_path)


def create_model_config(config: dict) -> SimpleNamespace:
    """Create model configuration namespace"""
    model_config = config.get("model", {})
    
    return SimpleNamespace(
        dmax=float(model_config.get("dmax", 10.0)),
        steps=int(model_config.get("steps", 8)),
        geometry=str(model_config.get("geometry", "hyper")),
        use_sparse=bool(model_config.get("use_sparse", True)),
        use_residual=bool(model_config.get("use_residual", True)),
        
        poisson_only=bool(model_config.get("poisson_only", False)),
        use_poisson=bool(model_config.get("use_poisson", True)),
        poisson_tol=float(model_config.get("poisson_tol", 1e-5)),
        poisson_maxiter=int(model_config.get("poisson_maxiter", 1000)),
        poisson_init=str(model_config.get("poisson_init", "est")),
        poisson_clip_to_max_gt=bool(model_config.get("poisson_clip_to_max_gt", False)),
        poisson_auto_flip=bool(model_config.get("poisson_auto_flip", True)),
        poisson_est_affine=bool(model_config.get("poisson_est_affine", True)),
        poisson_smooth_est=bool(model_config.get("poisson_smooth_est", True)),
        
        use_p_affinity=bool(model_config.get("use_p_affinity", True)),
        p_only_gate=bool(model_config.get("p_only_gate", False)),
        
        kernels=tuple(model_config.get("kernels", [3, 5])),
        kappa_min=float(model_config.get("kappa_min", 0.03)),
        kappa_max=float(model_config.get("kappa_max", 0.5)),
        
        anchor_learnable=bool(model_config.get("anchor_learnable", False)),
        anchor_mode=str(model_config.get("anchor_mode", "map")),
        anchor_alpha=float(model_config.get("anchor_alpha", 0.1)),
        
        min_gate=float(model_config.get("min_gate", 0.0)),
        min_alpha=float(model_config.get("min_alpha", 0.0))
    )


def get_trainer(dataset: str, model: OASIS_DC, device: torch.device, config: dict):
    """Get dataset-specific trainer"""
    if dataset.lower() == "nyu":
        return NYUTrainer(model, device, config)
    elif dataset.lower() == "kitti":
        return KITTITrainer(model, device, config)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser(description="Train OASIS-DC depth completion model")
    parser.add_argument("--dataset", type=str, required=True, choices=["nyu", "kitti"],
                        help="Dataset to use (nyu or kitti)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--shots", type=int, default=None,
                        help="Number of shots (overrides config file)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config file)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config["dataset"] = args.dataset
    config["save_dir"] = args.save_dir
    
    # Override shots and seed if provided
    data_config = config.get("data", {})
    if args.shots is not None:
        data_config["shots"] = args.shots
        print(f"Overriding shots to: {args.shots}")
    if args.seed is not None:
        data_config["seed"] = args.seed
        config["seed"] = args.seed
        print(f"Overriding seed to: {args.seed}")
    
    # Get shots and seed values
    shots = data_config.get("shots", 1)
    seed = data_config.get("seed", 0)
    
    # Apply shot-specific adaptations
    config = apply_shot_adaptations(config, shots)
    
    # Generate training list path if not explicitly set
    if "train_list" not in data_config:
        train_list = generate_train_list(args.dataset, data_config.get("data_root", ""), shots, seed)
        if train_list:
            data_config["train_list"] = train_list
            print(f"Using training list: {train_list}")
    
    # Update config with modified data section

    # Ensure training list file exists
    train_list_path = data_config.get("train_list")
    if train_list_path:
        data_root = data_config.get("data_root", "")
        if not ensure_train_list_exists(args.dataset, data_root, train_list_path, shots, seed):
            print(f"Failed to create or find training list: {train_list_path}")
            print("Please check your data_root path or create the training list manually.")
            sys.exit(1)
    config["data"] = data_config
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config}")
    
    # Create model
    model_cfg = create_model_config(config)
    model = OASIS_DC(model_cfg)
    
    # Get trainer
    trainer = get_trainer(args.dataset, model, device, config)
    
    # Setup optimizer
    train_config = config.get("train", {})
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config.get("lr", 1e-3),
        weight_decay=train_config.get("weight_decay", 1e-4)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = float('inf')
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, metrics = trainer.load_checkpoint(args.resume, optimizer)
        best_metric = metrics.get("rmse", float('inf'))
    
    # Training parameters
    epochs = train_config.get("epochs", 100)
    save_every = train_config.get("save_every", 10)
    eval_every = train_config.get("eval_every", 5)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get dataloaders
    train_loader = trainer.get_dataloader("train")
    val_loader = trainer.get_dataloader("val") if "val" in config.get("data", {}) else None
    
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Training loop
    for epoch in range(start_epoch + 1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, optimizer, epoch)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
        
        # Validation
        if val_loader and epoch % eval_every == 0:
            val_metrics = trainer.evaluate(val_loader, mode="full")
            print(f"Val - RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}")
            
            # Save best model
            current_metric = val_metrics['rmse']
            is_best = current_metric < best_metric
            if is_best:
                best_metric = current_metric
                print(f"New best validation RMSE: {best_metric:.4f}")
            
            # Save checkpoint
            if epoch % save_every == 0:
                checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch:03d}.pth")
                trainer.save_checkpoint(checkpoint_path, epoch, optimizer, val_metrics, is_best)
        
        elif epoch % save_every == 0:
            # Save checkpoint without validation
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            trainer.save_checkpoint(checkpoint_path, epoch, optimizer, train_metrics)
    
    # Final evaluation
    if val_loader:
        print("\n=== Final Evaluation ===")
        final_results = trainer.evaluate_all_modes(val_loader)
        trainer.print_results(final_results)
        
        # Save final results
        results_path = os.path.join(args.save_dir, "final_results.json")
        trainer.save_results(final_results, results_path)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()