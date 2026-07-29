#!/usr/bin/env python3
"""
Gate v18.6: Deep SAD-based OOS Detection
- Backbone: SmolLM2-1.7B (frozen or LoRA fine-tuned)
- Projection: Bias-free 2-layer MLP → 128d hypersphere
- Loss: Deep SAD (known attraction + OOS repulsion)
- Inference: threshold on ||feature - center|| for OOS detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import numpy as np


class DeepSADLoss(nn.Module):
    """
    Deep SAD loss: combine SVDD attraction (known→center) with OOS repulsion.
    
    Label convention (binary):
    - Known samples: label == 0 → minimize ||f - c||²  (attract to center)
    - OOS samples:  label == 1 → maximize ||f - c||² via hinge loss: max(0, margin² - ||f - c||²)
    
    This ensures known samples cluster tightly around center while OOS samples
    are pushed beyond the margin boundary.
    """

    def __init__(self, center: torch.Tensor, margin: float = 1.0, oos_weight: float = 0.5):
        """
        Args:
            center: [D] tensor, the hypersphere center (projected, 128d)
            margin: distance margin for OOS repulsion (default 1.0 for normalized features)
            oos_weight: weight for OOS repulsion loss (default 0.5)
        """
        super().__init__()
        self.register_buffer('center', center.float())
        self.margin = margin
        self.oos_weight = oos_weight

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Args:
            features: [N, D] normalized embeddings
            labels: [N] class labels; label == 0 = known, label == 1 = OOS
        Returns:
            dict with 'loss', 'known_loss', 'oos_loss', 'mean_known_dist', 'mean_oos_dist'
        """
        device = features.device
        center = self.center.to(device)
        
        # Compute squared L2 distance to center for all samples
        dist_sq = torch.sum((features - center) ** 2, dim=1)  # [N]
        
        # Known samples: attract to center (minimize distance)
        known_mask = labels == 0
        if known_mask.sum() > 0:
            known_dist_sq = dist_sq[known_mask]
            known_loss = known_dist_sq.mean()
            mean_known_dist = torch.sqrt(known_dist_sq).mean().item()
        else:
            known_loss = torch.tensor(0.0, device=device)
            mean_known_dist = 0.0
        
        # OOS samples: repel from center (push beyond margin)
        oos_mask = labels == 1
        if oos_mask.sum() > 0:
            oos_dist_sq = dist_sq[oos_mask]
            # Hinge loss: penalize if distance < margin
            margin_sq = self.margin ** 2
            oos_loss = torch.clamp(margin_sq - oos_dist_sq, min=0.0).mean()
            mean_oos_dist = torch.sqrt(oos_dist_sq).mean().item()
        else:
            oos_loss = torch.tensor(0.0, device=device)
            mean_oos_dist = 0.0
        
        # Combined loss
        total_loss = known_loss + self.oos_weight * oos_loss
        
        return {
            'loss': total_loss,
            'known_loss': known_loss.item() if torch.is_tensor(known_loss) else known_loss,
            'oos_loss': oos_loss.item() if torch.is_tensor(oos_loss) else oos_loss,
            'mean_known_dist': mean_known_dist,
            'mean_oos_dist': mean_oos_dist,
        }


class SVDDLoss(nn.Module):
    """Deep SVDD loss: minimize hypersphere radius around center for known samples.
    
    DEPRECATED: Use DeepSADLoss for proper OOS repulsion.
    """

    def __init__(self, center: torch.Tensor, nu: float = 0.1):
        """
        Args:
            center: [D] tensor, the hypersphere center (e.g., global ID center)
            nu: soft-boundary parameter (0.1 = allow 10% outliers)
        """
        super().__init__()
        self.register_buffer('center', center.float())
        self.nu = nu

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, D] normalized embeddings
            labels: [N] class labels; only samples with label >= 0 contribute to loss
        Returns:
            scalar loss
        """
        device = features.device
        center = self.center.to(device)
        
        # Only compute loss on known samples (label >= 0)
        known_mask = labels >= 0
        if known_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        known_features = features[known_mask]
        # Compute squared L2 distance to center
        dist_sq = torch.sum((known_features - center) ** 2, dim=1)
        
        # Soft-boundary: allow nu fraction to be outside
        loss = dist_sq.mean()
        return loss


class SVDDGate(nn.Module):
    """
    Gate for OOS detection using Deep SAD.
    - Encodes input text into 128d hypersphere
    - Computes distance to center in projected space
    - Decision: dist < threshold → Known; else → OOS
    """

    def __init__(
        self,
        model_path: str,
        center_path: str = 'models/gate_center_v18.7.npy',
        projection_dim: int = 128,
        freeze_backbone: bool = True,
        lora_r: int = 0,
        lora_alpha: int = 0,
        projection_weights_path: str = None,
    ) -> None:
        """
        Args:
            model_path: path to SmolLM2-1.7B backbone
            center_path: path to center .npy (projected 128d preferred)
            projection_dim: projection head output dimension
            freeze_backbone: if True, freeze backbone and only train projection
            lora_r: if >0, use LoRA with this rank (overrides freeze_backbone)
            lora_alpha: LoRA alpha scaling
            projection_weights_path: optional path to load projection head weights
                                     (for consistency with center initialization)
        """
        super().__init__()

        self.base = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        hidden_size = getattr(self.base.config, 'hidden_size', None)
        if hidden_size is None:
            raise RuntimeError('Base model missing hidden_size in config')

        if lora_r > 0:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias='none',
                task_type='FEATURE_EXTRACTION'
            )
            self.base = get_peft_model(self.base, lora_cfg)
            freeze_backbone = False

        if freeze_backbone:
            for param in self.base.parameters():
                param.requires_grad = False

        from legacy.models.components import ProjectionHead
        self.projection = ProjectionHead(hidden_size, projection_dim=projection_dim, bias=False)
        
        # Load projection weights if provided (for center consistency)
        if projection_weights_path is not None:
            import os
            if os.path.exists(projection_weights_path):
                self.projection.load_state_dict(torch.load(projection_weights_path, map_location='cpu'))
                # Freeze projection if loading pre-computed weights (optional, can unfreeze)
                # for param in self.projection.parameters():
                #     param.requires_grad = False
            else:
                import logging
                logging.warning('projection_weights_path not found: %s, using random init', projection_weights_path)

        center_raw = np.load(center_path)
        center_tensor = torch.from_numpy(center_raw).float()

        if center_tensor.numel() == projection_dim:
            self.register_buffer('center_projected', F.normalize(center_tensor, dim=0))
            self.register_buffer('center_raw', torch.zeros(hidden_size))
            self.center_initialized = True
        elif center_tensor.numel() == hidden_size:
            self.register_buffer('center_raw', center_tensor)
            self.register_buffer('center_projected', torch.zeros(projection_dim))
            self.center_initialized = False
        else:
            raise RuntimeError(
                f'Center dimension mismatch: got {center_tensor.numel()}, '
                f'expected {projection_dim} (projected) or {hidden_size} (raw)'
            )

    def _initialize_projected_center(self) -> None:
        """Compute projected center by passing raw center through projection head."""
        if self.center_initialized:
            return
        with torch.no_grad():
            center_proj = self.projection(self.center_raw.unsqueeze(0))
            self.center_projected = F.normalize(center_proj.squeeze(0), dim=0)
            self.center_initialized = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
        Returns:
            features: [B, projection_dim] normalized embeddings
        """
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
            hidden = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        else:
            raise RuntimeError('Model output missing hidden states')

        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        sum_embeddings = torch.sum(hidden * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        proj = self.projection(pooled)
        features = F.normalize(proj, dim=1)
        return features

    def compute_distances(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 distance from features to projected center.
        Args:
            features: [B, projection_dim] normalized embeddings
        Returns:
            distances: [B] L2 distances
        """
        if not self.center_initialized:
            self._initialize_projected_center()
        center = self.center_projected.to(features.device)
        return torch.norm(features - center, p=2, dim=1)

    def predict_oos(self, features: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Predict OOS based on distance threshold.
        Args:
            features: [B, projection_dim]
            threshold: distance threshold (determined from validation set)
        Returns:
            is_oos: [B] boolean tensor (True = OOS)
        """
        dist = self.compute_distances(features)
        return dist > threshold
