"""
NYU Depth V2 specific trainer.
"""
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

from src.trainers.base_trainer import BaseTrainer
from src.datasets.nyu_dataset import build_nyu_dataloaders


class NYUTrainer(BaseTrainer):
    """NYU Depth V2 specific trainer"""
    
    def __init__(self, model, device: torch.device, config: Dict[str, Any]):
        super().__init__(model, device, config)
        
        # NYU-specific config defaults
        self.config.setdefault("dmax", 10.0)
        self.config.setdefault("target_size", [240, 320])
        self.config.setdefault("n_sparse", 500)
        self.config.setdefault("est_mode", "precomputed")
        self.config.setdefault("pre_mono_dir", "mono_rel")
    
    def get_dataloader(self, split: str, **kwargs) -> DataLoader:
        """Get NYU dataloader for specified split"""
        if split == "train":
            train_loader, _ = build_nyu_dataloaders(
                root_dir=self.config["data_root"],
                train_list=self.config["train_list"],
                test_list=self.config.get("test_list", self.config["train_list"]),  # dummy for train
                n_sparse=self.config["n_sparse"],
                target_size=tuple(self.config["target_size"]),
                est_mode=self.config["est_mode"],
                pre_mono_dir=self.config["pre_mono_dir"],
                batch_size=self.config.get("batch_size", 1),
                num_workers=self.config.get("num_workers", 4),
                seed=self.config.get("seed", 0),
                **kwargs
            )
            return train_loader
            
        elif split in ["test", "val"]:
            _, test_loader = build_nyu_dataloaders(
                root_dir=self.config["data_root"],
                train_list=self.config.get("train_list", self.config["test_list"]),  # dummy for test
                test_list=self.config["test_list"], 
                n_sparse=self.config["n_sparse"],
                target_size=tuple(self.config["target_size"]),
                est_mode=self.config["est_mode"],
                pre_mono_dir=self.config["pre_mono_dir"],
                batch_size=self.config.get("batch_size", 1),
                num_workers=self.config.get("num_workers", 4),
                seed=self.config.get("seed", 0),
                **kwargs
            )
            return test_loader
            
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """NYU-specific batch preprocessing"""
        # NYU data is already properly formatted from dataset
        return {
            "rgb": batch["rgb"],
            "sparse_depth": batch["sparse_depth"], 
            "sparse_mask": batch["sparse_mask"],
            "est_depth_norm": batch["est_depth_norm"],
            "gt_depth": batch["gt_depth"],
            "valid_mask": batch["valid_mask"]
        }
    
    def train_one_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int) -> Dict[str, float]:
        """NYU-specific training for one epoch"""
        return self.train_epoch(train_loader, optimizer, epoch)
    
    def evaluate_all_modes(self, test_loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """Evaluate all modes: full, residual_off, poisson_only"""
        results = {}
        
        print("Evaluating full model...")
        results["full"] = self.evaluate(test_loader, mode="full")
        
        print("Evaluating residual-off mode...")
        results["residual_off"] = self.evaluate(test_loader, mode="residual_off")
        
        print("Evaluating Poisson-only mode...")
        results["poisson_only"] = self.evaluate(test_loader, mode="poisson_only")
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print NYU evaluation results"""
        print("\n=== NYU Depth V2 Results ===")
        
        if isinstance(results, dict) and "full" in results:
            # Multi-mode results
            for mode, metrics in results.items():
                print(f"\n[{mode.upper()}]")
                print(f"  RMSE: {metrics['rmse']:.4f} m")
                print(f"  MAE:  {metrics['mae']:.4f} m") 
                print(f"  δ1:   {metrics['delta1']:.4f}")
                
                if "avg_solver_time" in metrics:
                    print(f"  Solver time: {metrics['avg_solver_time']:.3f}s")
        else:
            # Single mode results
            print(f"  RMSE: {results['rmse']:.4f} m")
            print(f"  MAE:  {results['mae']:.4f} m")
            print(f"  δ1:   {results['delta1']:.4f}")
            
            if "avg_solver_time" in results:
                print(f"  Solver time: {results['avg_solver_time']:.3f}s")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get NYU-specific model configuration"""
        return {
            "dmax": self.config["dmax"],
            "steps": self.config.get("steps", 8),
            "geometry": self.config.get("geometry", "hyper"),
            "use_sparse": self.config.get("use_sparse", True),
            "use_residual": self.config.get("use_residual", True),
            "poisson_only": False,
            "use_poisson": True,
            "poisson_tol": self.config.get("poisson_tol", 1e-5),
            "poisson_maxiter": self.config.get("poisson_maxiter", 1000),
            "poisson_init": self.config.get("poisson_init", "est"),
            "poisson_clip_to_max_gt": False,
            "poisson_auto_flip": self.config.get("poisson_auto_flip", True),
            "poisson_est_affine": self.config.get("poisson_est_affine", True), 
            "poisson_smooth_est": self.config.get("poisson_smooth_est", True),
            "use_p_affinity": self.config.get("use_p_affinity", True),
            "p_only_gate": self.config.get("p_only_gate", False),
            "kernels": self.config.get("kernels", [3, 5]),
            "kappa_min": self.config.get("kappa_min", 0.03),
            "kappa_max": self.config.get("kappa_max", 0.5),
            "anchor_learnable": self.config.get("anchor_learnable", False),
            "anchor_mode": self.config.get("anchor_mode", "map"),
            "anchor_alpha": self.config.get("anchor_alpha", 0.1),
            "min_gate": self.config.get("min_gate", 0.0),
            "min_alpha": self.config.get("min_alpha", 0.0)
        }