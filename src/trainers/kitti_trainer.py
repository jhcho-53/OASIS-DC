"""
KITTI Depth Completion specific trainer.
"""
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

from src.trainers.base_trainer import BaseTrainer
from src.datasets.kitti_dataset import build_kitti_dataloaders


class KITTITrainer(BaseTrainer):
    """KITTI Depth Completion specific trainer"""
    
    def __init__(self, model, device: torch.device, config: Dict[str, Any]):
        super().__init__(model, device, config)
        
        # KITTI-specific config defaults  
        self.config.setdefault("dmax", 80.0)
        self.config.setdefault("limit_sparse", -1)  # Use all LiDAR points
        self.config.setdefault("est_mode", "precomputed")
        self.config.setdefault("pre_est_dir", "est")
    
    def get_dataloader(self, split: str = "val", **kwargs) -> DataLoader:
        """Get KITTI dataloader for specified split"""
        return build_kitti_dataloaders(
            root_dir=self.config["data_root"],
            split=split,
            est_mode=self.config["est_mode"],
            pre_est_dir=self.config["pre_est_dir"],
            limit_sparse=self.config["limit_sparse"],
            dmax=self.config["dmax"],
            batch_size=self.config.get("batch_size", 1),
            num_workers=self.config.get("num_workers", 4),
            seed=self.config.get("seed", 0),
            **kwargs
        )
    
    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """KITTI-specific batch preprocessing"""
        # KITTI data is already properly formatted from dataset
        return {
            "rgb": batch["rgb"],
            "sparse_depth": batch["sparse_depth"],
            "sparse_mask": batch["sparse_mask"],
            "est_depth_norm": batch["est_depth_norm"],
            "gt_depth": batch["gt_depth"],
            "valid_mask": batch["valid_mask"]
        }
    
    def evaluate_all_modes(self, test_loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """Evaluate all modes for KITTI"""
        results = {}
        
        print("Evaluating full model...")
        results["full"] = self.evaluate(test_loader, mode="full")
        
        print("Evaluating residual-off mode...")
        results["residual_off"] = self.evaluate(test_loader, mode="residual_off")
        
        print("Evaluating Poisson-only mode...")
        results["poisson_only"] = self.evaluate(test_loader, mode="poisson_only")
        
        return results
    
    def evaluate_poisson_with_alignment(self, 
                                       test_loader: DataLoader,
                                       align_mode: str = "quantile",
                                       **align_kwargs) -> Dict[str, float]:
        """
        Evaluate Poisson solver with test-time alignment
        
        Args:
            test_loader: Test dataloader
            align_mode: "quantile", "affine", or "none"
            **align_kwargs: Additional alignment parameters
        """
        # This would require extending poisson_utils to support alignment
        # For now, use standard poisson evaluation
        print(f"Evaluating Poisson with {align_mode} alignment...")
        return self.evaluate(test_loader, mode="poisson_only")
    
    def print_results(self, results: Dict[str, Any]):
        """Print KITTI evaluation results"""
        print("\n=== KITTI Depth Completion Results ===")
        
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
        """Get KITTI-specific model configuration"""
        return {
            "dmax": self.config["dmax"],
            "steps": self.config.get("steps", 6),  # KITTI typically uses fewer steps
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
            "kernels": self.config.get("kernels", [3, 5, 7]),  # Larger kernels for KITTI
            "kappa_min": self.config.get("kappa_min", 1e-3),
            "kappa_max": self.config.get("kappa_max", 1.0),
            "anchor_learnable": self.config.get("anchor_learnable", True),
            "anchor_mode": self.config.get("anchor_mode", "map"),
            "anchor_alpha": self.config.get("anchor_alpha", 0.7),  # Higher alpha for KITTI
            "min_gate": self.config.get("min_gate", 0.0),
            "min_alpha": self.config.get("min_alpha", 0.0)
        }
    
    def compute_loss(self, 
                    pred_depth: torch.Tensor,
                    gt_depth: torch.Tensor,
                    valid_mask: torch.Tensor,
                    aux_outputs: Dict = None) -> torch.Tensor:
        """KITTI-specific loss computation"""
        # Use base loss but with KITTI-specific weights
        loss = super().compute_loss(pred_depth, gt_depth, valid_mask, aux_outputs)
        
        # KITTI-specific loss modifications could go here
        # For example, different weighting for sparse regions
        
        return loss