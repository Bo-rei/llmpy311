"""
Multi-center SVDD Loss
======================
Soft-boundary Deep SVDD loss 的多中心变体

核心修改：
- 单中心：hinge_loss = max(0, dist² - R²)
- 多中心：hinge_loss = max(0, min_dist² - R²)
  其中 min_dist = min_k ||φ(x) - c_k||

L2 正则化仅对 centers 参数
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class MultiCenterSVDDLoss(nn.Module):
    """
    Soft-boundary Multi-center Deep SVDD Loss
    
    L = R² + (1/νn)Σ[max(0, min_dist²(x) - R²)] + (λ/2)||centers||²_F
    
    与单中心唯一区别：距离计算方式
    """
    
    def __init__(self, nu: float = 0.1, lambda_reg: float = 1e-5):
        """
        Args:
            nu: 允许的训练集 slack 比例（目标 outlier 比例）
            lambda_reg: centers 的 L2 正则化系数
        """
        super().__init__()
        self.nu = nu
        self.lambda_reg = lambda_reg
    
    def forward(
        self,
        min_distances: torch.Tensor,
        centers: torch.Tensor,
        R: float = None
    ) -> dict:
        """
        计算 multi-center SVDD loss
        
        Args:
            min_distances: (N,) - 每个样本到最近中心的距离
            centers: (K, D) - 所有中心（用于正则化）
            R: 当前半径（若 None 则从 quantile 计算）
        
        Returns:
            dict with keys:
                - loss: 总损失
                - radius: R
                - hinge_loss: (1/νn)Σmax(0, min_dist² - R²)
                - reg_loss: (λ/2)||centers||²_F
                - slack_ratio: (min_dist² > R²) 的样本比例
        """
        N = len(min_distances)
        
        # Step 1: 确定 R（从 (1-ν) quantile）
        if R is None:
            dist_squared = min_distances ** 2
            R_squared = torch.quantile(dist_squared, 1 - self.nu)
            R = torch.sqrt(R_squared)
        else:
            R_squared = R ** 2
        
        # Step 2: Hinge loss（违反半径约束的惩罚）
        dist_squared = min_distances ** 2
        slack = torch.clamp(dist_squared - R_squared, min=0.0)  # max(0, dist² - R²)
        hinge_loss = slack.sum() / (self.nu * N)
        
        # Step 3: L2 正则化（所有 centers）
        reg_loss = 0.5 * self.lambda_reg * torch.sum(centers ** 2)
        
        # Step 4: 总损失
        total_loss = R_squared + hinge_loss + reg_loss
        
        # Step 5: 统计 slack ratio（应该 ≈ ν）
        slack_ratio = (dist_squared > R_squared).float().mean().item()
        
        return {
            'loss': total_loss,
            'radius': R.item(),
            'hinge_loss': hinge_loss.item(),
            'reg_loss': reg_loss.item(),
            'slack_ratio': slack_ratio
        }
    
    def compute_frr(
        self,
        known_distances: torch.Tensor,
        R: float
    ) -> float:
        """
        计算 False Reject Rate（Known 样本被拒绝的比例）
        
        Args:
            known_distances: (N_known,) - Known 样本到最近中心的距离
            R: 决策半径
        
        Returns:
            frr: Known 样本中 distance > R 的比例
        """
        reject_count = (known_distances > R).sum().item()
        total_count = len(known_distances)
        frr = reject_count / total_count if total_count > 0 else 0.0
        return frr


class MultiCenterSVDDGateLoss(nn.Module):
    """
    Multi-center Deep SVDD Loss (soft-boundary) + Center Repulsion

    L = R² + (1/(νN)) Σ max(0, min_dist_sq - R²) + λ_rep * L_rep
    L_rep = mean_{i≠j} exp(-||c_i - c_j||²)
    """

    def __init__(self, nu: float = 0.1, lambda_rep: float = 0.01, init_radius: float = 1.0):
        super().__init__()
        if not 0 < nu < 1:
            raise ValueError(f"ν 必须在 (0,1) 之间，当前: {nu}")

        self.nu = nu
        self.lambda_rep = lambda_rep
        self.radius_squared = nn.Parameter(torch.tensor(float(init_radius) ** 2))

        logger.info("MultiCenterSVDDGateLoss: ν=%.2f, λ_rep=%.4f, R²_init=%.4f",
                    nu, lambda_rep, float(init_radius) ** 2)

    def forward(self, min_dist_sq: torch.Tensor, centers: torch.Tensor) -> dict:
        N = min_dist_sq.shape[0]

        r2 = torch.clamp(self.radius_squared, min=1e-8)
        slack = torch.clamp(min_dist_sq - r2, min=0.0)
        hinge_loss = slack.sum() / (self.nu * N)

        # Repulsion among centers: -1 / (dist² + ε) to strongly push apart
        center_dist_sq = torch.cdist(centers, centers, p=2) ** 2
        mask = ~torch.eye(center_dist_sq.size(0), dtype=torch.bool, device=center_dist_sq.device)
        # Higher repulsion = minimize -1/dist² = maximize dist²
        repulsion = -1.0 / (center_dist_sq[mask] + 1e-6).mean()

        total_loss = r2 + hinge_loss + self.lambda_rep * repulsion

        slack_ratio = (min_dist_sq > r2).float().mean()

        return {
            "loss": total_loss,
            "radius_squared": r2.detach(),
            "hinge_loss": hinge_loss.detach(),
            "repulsion_loss": repulsion.detach(),
            "slack_ratio": slack_ratio.detach()
        }
