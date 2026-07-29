#!/usr/bin/env python3
"""
TASK-202: SVDD Head with Anti-Collapse Mechanisms

Implements Deep SVDD projection head with:
1. No bias in Linear layers (hard constraint 1)
2. Unbounded activation (LeakyReLU, hard constraint 2)
3. Variance Regularization (prevent subspace collapse)
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class VarianceRegularization(nn.Module):
    """
    Compute negative trace of covariance matrix (negative total variance).
    Used as a regularization term to prevent subspace collapse.
    
    Loss = -trace(Cov(Z)) where Cov(Z) = E[(z - mean(z))^T(z - mean(z))]
    Minimizing -trace(Cov) = Maximizing trace(Cov) = Maximizing total variance
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, Z):
        """
        Args:
            Z: (batch_size, d_proj) tensor of projected features
        
        Returns:
            scalar: -trace(Cov(Z)) = negative total variance
        """
        # Center the features
        Z_mean = Z.mean(dim=0, keepdim=True)  # (1, d_proj)
        Z_centered = Z - Z_mean  # (batch_size, d_proj)
        
        # Compute covariance matrix (unnormalized)
        cov_matrix = (Z_centered.t() @ Z_centered) / Z.size(0)  # (d_proj, d_proj)
        
        # Trace is sum of diagonal elements
        trace = torch.trace(cov_matrix)
        
        # Return negative trace (to be minimized as a loss term)
        return -trace


class SVDDHead(nn.Module):
    """
    Deep SVDD Projection Head with anti-collapse constraints.
    
    Architecture:
        input (d_model) -> Linear(d_model, d_proj, bias=False) -> LeakyReLU -> 
        Linear(d_proj, d_proj, bias=False) -> output (d_proj)
    
    Constraints:
    1. bias=False in all Linear layers
    2. Unbounded activation (LeakyReLU)
    3. Fixed center C (external parameter)
    """
    
    def __init__(self, d_model, d_proj=256):
        super().__init__()
        
        self.d_model = d_model
        self.d_proj = d_proj
        
        # Layer 1: Linear (no bias) -> LeakyReLU -> Linear (no bias)
        self.proj1 = nn.Linear(d_model, d_proj, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.proj2 = nn.Linear(d_proj, d_proj, bias=False)
        
        # Verify no bias
        assert self.proj1.bias is None, "proj1 must have bias=False"
        assert self.proj2.bias is None, "proj2 must have bias=False"
        
        logger.info(f"SVDDHead initialized: {d_model} -> {d_proj} -> {d_proj}")
    
    def forward(self, h):
        """
        Forward pass.
        
        Args:
            h: (batch_size, d_model) representation from backbone
        
        Returns:
            z: (batch_size, d_proj) projected features
        """
        z = self.proj1(h)
        z = self.activation(z)
        z = self.proj2(z)
        return z


class SVDDLoss(nn.Module):
    """
    Deep SVDD Loss with Variance Regularization.
    
    L = mean(||z - C||^2) + lambda_1 * ||W||_F^2 + lambda_2 * (-trace(Cov(Z)))
    """
    
    def __init__(self, svdd_head, lambda_1=1e-4, lambda_2=1e-3):
        super().__init__()
        
        self.svdd_head = svdd_head
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.variance_reg = VarianceRegularization()
    
    def forward(self, h, C):
        """
        Compute SVDD loss.
        
        Args:
            h: (batch_size, d_model) hidden states from backbone
            C: (d_proj,) fixed center, pre-computed
        
        Returns:
            loss: scalar
            breakdown: dict with loss components
        """
        # Project features
        z = self.svdd_head(h)
        
        # Distance loss (Deep SVDD)
        dist = torch.norm(z - C, p=2, dim=1)  # (batch_size,)
        loss_svdd = dist.mean()
        
        # Weight decay
        weight_norm = sum(p.norm()**2 for p in self.svdd_head.parameters())
        loss_wd = self.lambda_1 * weight_norm
        
        # Variance regularization
        loss_var = self.lambda_2 * self.variance_reg(z)
        
        # Total loss
        total_loss = loss_svdd + loss_wd + loss_var
        
        breakdown = {
            'loss_svdd': loss_svdd.item(),
            'loss_wd': loss_wd.item(),
            'loss_var': loss_var.item(),
            'total': total_loss.item()
        }
        
        return total_loss, breakdown
