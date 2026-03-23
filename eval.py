#!/usr/bin/env python3
"""
Unified evaluation script for OASIS-DC depth completion.
"""
import os
import sys
import argparse
import yaml
from types import SimpleNamespace

import torch

from models.model import OASIS_DC
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
    parser = argparse.ArgumentParser(description="Evaluate OASIS-DC depth completion model")
    parser.add_argument("--dataset", type=str, required=True, choices=["nyu", "kitti"],
                        help="Dataset to use (nyu or kitti)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (optional for Poisson-only)")
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["all", "full", "residual_off", "poisson_only"],
                        help="Evaluation mode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Output file for results")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save prediction images")
    parser.add_argument("--pred-dir", type=str, default="predictions",
                        help="Directory to save predictions")
    parser.add_argument("--shots", type=int, default=None,
                        help="Number of shots (for n-shot configs)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config["dataset"] = args.dataset
    
    # Apply shot adaptations if shots parameter is provided
    if args.shots is not None:
        config = apply_shot_adaptations(config, args.shots)
        print(f"Evaluating with {args.shots}-shot configuration")
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config}")
    print(f"Evaluation mode: {args.mode}")
    
    # Create model
    model_cfg = create_model_config(config)
    model = OASIS_DC(model_cfg)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    elif args.mode in ["full", "residual_off"]:
        print("Warning: No checkpoint provided but mode requires trained model")
    
    # Get trainer
    trainer = get_trainer(args.dataset, model, device, config)
    
    # Get test dataloader
    test_loader = trainer.get_dataloader("test")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Run evaluation
    print("\n=== Starting Evaluation ===")
    
    if args.mode == "all":
        # Evaluate all modes
        results = trainer.evaluate_all_modes(test_loader)
        
    else:
        # Evaluate specific mode
        results = trainer.evaluate(test_loader, mode=args.mode)
    
    # Print results
    trainer.print_results(results)
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    trainer.save_results(results, args.output)
    
    # Save predictions if requested
    if args.save_predictions:
        print(f"Prediction saving not yet implemented")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()