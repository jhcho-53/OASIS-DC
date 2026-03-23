"""
Base trainer class for depth completion tasks.
"""
import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import OASIS_DC
from src.utils.metrics import compute_all_metrics, format_metrics
from src.utils.poisson_utils import solve_poisson_with_stats


class BaseTrainer(ABC):
    """Base trainer class with common training/evaluation logic"""
    
    def __init__(self, 
                 model: OASIS_DC, 
                 device: torch.device,
                 config: Dict[str, Any]):
        self.model = model
        self.device = device
        self.config = config
        self.model.to(device)
        
    @abstractmethod
    def get_dataloader(self, split: str, **kwargs) -> DataLoader:
        """Get dataset-specific dataloader"""
        pass
    
    @abstractmethod
    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Dataset-specific batch preprocessing"""
        pass
    
    def train_epoch(self, 
                   train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # Preprocess batch
            processed_batch = self.preprocess_batch(batch)
            
            # Forward pass
            I = processed_batch["rgb"].to(self.device)
            DL = processed_batch["sparse_depth"].to(self.device) 
            ML = processed_batch["sparse_mask"].to(self.device)
            E_norm = processed_batch["est_depth_norm"].to(self.device)
            GT = processed_batch["gt_depth"].to(self.device)
            valid_mask = processed_batch["valid_mask"].to(self.device)
            
            pred_depth, aux_outputs = self.model(I, DL, ML, E_norm)
            
            # Compute loss
            loss = self.compute_loss(pred_depth, GT, valid_mask, aux_outputs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                batch_metrics = compute_all_metrics(pred_depth, GT, valid_mask)
                batch_metrics["loss"] = float(loss.item())
                
                for k, v in batch_metrics.items():
                    epoch_metrics[k] += v
                
                num_batches += 1
                
                # Update progress bar
                if batch_idx % 10 == 0:
                    current_metrics = {k: v/num_batches for k, v in epoch_metrics.items()}
                    pbar.set_postfix(format_metrics(current_metrics, precision=4))
        
        # Average metrics over epoch
        return {k: v/num_batches for k, v in epoch_metrics.items()}
    
    @torch.no_grad()
    def evaluate(self, 
                 eval_loader: DataLoader,
                 mode: str = "full") -> Dict[str, float]:
        """
        Evaluate model
        
        Args:
            eval_loader: Evaluation dataloader
            mode: Evaluation mode ("full", "residual_off", "poisson_only")
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        epoch_metrics = defaultdict(float)
        num_samples = 0
        solver_times = []
        
        pbar = tqdm(eval_loader, desc=f"Eval ({mode})", ncols=100)
        
        for batch in pbar:
            # Preprocess batch
            processed_batch = self.preprocess_batch(batch)
            
            I = processed_batch["rgb"].to(self.device)
            DL = processed_batch["sparse_depth"].to(self.device)
            ML = processed_batch["sparse_mask"].to(self.device) 
            E_norm = processed_batch["est_depth_norm"].to(self.device)
            GT = processed_batch["gt_depth"].to(self.device)
            valid_mask = processed_batch["valid_mask"].to(self.device)
            
            if mode == "poisson_only":
                # Evaluate Poisson solver only
                pred_depth, solver_stats = solve_poisson_with_stats(
                    sparse_depth=DL,
                    mask_sparse=ML.bool().float(),
                    est_depth=E_norm,
                    dmax=self.config.get("dmax", 10.0),
                    tolerance=self.config.get("poisson_tol", 1e-5),
                    max_iterations=self.config.get("poisson_maxiter", 1000),
                    device=str(self.device)
                )
                solver_times.append(solver_stats.get("time_sec", 0.0))
                
            elif mode == "residual_off":
                # Disable residual refinement
                if hasattr(self.model, "res"):
                    self.model.res = None
                pred_depth, aux_outputs = self.model(I, DL, ML, E_norm)
                
            else:  # mode == "full"
                # Full model evaluation
                pred_depth, aux_outputs = self.model(I, DL, ML, E_norm)
            
            # Compute metrics
            batch_metrics = compute_all_metrics(pred_depth, GT, valid_mask)
            
            for k, v in batch_metrics.items():
                epoch_metrics[k] += v
            num_samples += 1
            
            # Update progress bar
            if num_samples % 10 == 0:
                current_metrics = {k: v/num_samples for k, v in epoch_metrics.items()}
                pbar.set_postfix(format_metrics(current_metrics, precision=4))
        
        # Average metrics
        final_metrics = {k: v/num_samples for k, v in epoch_metrics.items()}
        
        # Add solver timing if available
        if solver_times:
            final_metrics["avg_solver_time"] = float(sum(solver_times) / len(solver_times))
            final_metrics["total_solver_time"] = float(sum(solver_times))
        
        return final_metrics
    
    def compute_loss(self, 
                    pred_depth: torch.Tensor,
                    gt_depth: torch.Tensor, 
                    valid_mask: torch.Tensor,
                    aux_outputs: Optional[Dict] = None) -> torch.Tensor:
        """
        Compute training loss
        
        Args:
            pred_depth: Predicted depth
            gt_depth: Ground truth depth
            valid_mask: Valid pixel mask
            aux_outputs: Auxiliary model outputs
        
        Returns:
            Loss tensor
        """
        # Basic L1 + scale-invariant loss
        valid = valid_mask > 0
        
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred_depth.device, requires_grad=True)
        
        pred_valid = pred_depth[valid]
        gt_valid = gt_depth[valid]
        
        # L1 loss
        l1_loss = torch.mean(torch.abs(pred_valid - gt_valid))
        
        # Scale-invariant loss
        log_pred = torch.log(torch.clamp(pred_valid, min=0.1))
        log_gt = torch.log(torch.clamp(gt_valid, min=0.1))
        log_diff = log_pred - log_gt
        
        si_loss = torch.mean(log_diff ** 2) - (self.config.get("mu_scaleinv", 0.1) * 
                                              (torch.mean(log_diff) ** 2))
        
        total_loss = l1_loss + si_loss
        
        # Add auxiliary losses if available
        if aux_outputs and "anchor_loss" in aux_outputs:
            anchor_weight = self.config.get("w_anchor_reg", 0.0)
            total_loss += anchor_weight * aux_outputs["anchor_loss"]
        
        return total_loss
    
    def save_checkpoint(self, 
                       checkpoint_path: str,
                       epoch: int,
                       optimizer: torch.optim.Optimizer,
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = checkpoint_path.replace(".pth", "_best.pth")
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint["epoch"], checkpoint.get("metrics", {})
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Make results JSON serializable
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, (torch.Tensor, torch.device)):
                serializable_results[k] = str(v)
            elif isinstance(v, (int, float, str, bool, list, dict)):
                serializable_results[k] = v
            else:
                serializable_results[k] = str(v)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {output_path}")