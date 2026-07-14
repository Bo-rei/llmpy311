"""
Soft-boundary Deep SVDD Loss

严格实现 ICML 2018 论文 Eq. (3):

    min_{R,W} R² + (1/νn) Σ max(0, ||φ(x_i;W) - c||² - R²) + (λ/2) Σ ||W^ℓ||²_F

where:
    - R: soft boundary radius (从 (1-ν) 分位数计算，不参与梯度)
    - ν: 允许的 Known 离群比例 (如 0.1 = 10%)
    - c: center (固定，不参与训练)
    - φ(x;W): MLP encoder 输出
    - λ: L2 正则化系数
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SVDDLoss(nn.Module):
    """
    Soft-boundary Deep SVDD Loss (ICML 2018)
    
    论文公式分解:
        L = L_radius + L_hinge + L_reg
        
        L_radius = R²                                              (体积最小化)
        L_hinge  = (1/νn) Σ max(0, dist²_i - R²)                 (软边界惩罚)
        L_reg    = (λ/2) Σ ||W||²_F                               (L2 正则化)
    
    Args:
        nu: 允许的 Known 离群比例 (default: 0.1 = 10%)
        lambda_reg: L2 正则化系数 (default: 1e-6)
        use_radius_loss: 是否在训练 loss 中包含 R² 项 (default: True)
    """
    
    def __init__(
        self,
        nu: float = 0.1,
        lambda_reg: float = 1e-6,
        use_radius_loss: bool = True
    ):
        super().__init__()
        
        if not 0 < nu < 1:
            raise ValueError(f"ν 必须在 (0,1) 之间，当前: {nu}")
        
        self.nu = nu
        self.lambda_reg = lambda_reg
        self.use_radius_loss = use_radius_loss
        
        logger.info(f"SVDD Loss 配置: ν={nu:.2f}, λ={lambda_reg:.2e}, use_R²={use_radius_loss}")
    
    def forward(
        self,
        distances_squared: torch.Tensor,
        model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        计算 Soft-boundary Deep SVDD Loss
        
        Args:
            distances_squared: (n,) - ||φ(x_i) - c||² for all samples
            model: Gate 模型 (用于计算 L2 正则化)
        
        Returns:
            {
                'loss': total loss,
                'radius': R (从分位数计算),
                'radius_squared': R²,
                'radius_loss': R² (论文第一项),
                'hinge_loss': (1/νn) Σ slack (论文第二项),
                'reg_loss': (λ/2) Σ ||W||² (论文第三项),
                'slack_ratio': slack > 0 的比例 (应 ≈ ν),
                'mean_distance': 平均距离 (用于监控)
            }
        """
        n = distances_squared.size(0)
        
        # ========== 第1项: 半径 R² (体积最小化) ==========
        # R = quantile(dist², 1-ν)  (论文 Eq. 3)
        # 例如 ν=0.1 → 取 90th 百分位作为半径
        quantile_value = 1.0 - self.nu
        
        with torch.no_grad():
            radius_squared = torch.quantile(
                distances_squared.detach(),  # 不参与梯度！
                quantile_value
            )
            radius = torch.sqrt(radius_squared + 1e-8)
        
        # R² 项（论文公式第一项）
        if self.use_radius_loss:
            radius_loss = radius_squared
        else:
            radius_loss = torch.tensor(0.0, device=distances_squared.device)
        
        # ========== 第2项: Hinge Loss (软边界惩罚) ==========
        # (1/νn) Σ max(0, dist²_i - R²)
        slack = torch.relu(distances_squared - radius_squared)  # max(0, ...)
        hinge_loss = slack.sum() / (self.nu * n)  # (1/νn) Σ
        
        # 统计 slack > 0 的比例（理论上应 ≈ ν）
        slack_ratio = (slack > 0).float().mean().item()
        
        # ========== 第3项: L2 正则化 (权重衰减) ==========
        # (λ/2) Σ ||W^ℓ||²_F
        reg_loss = 0.0
        for param in model.parameters():
            if param.requires_grad:
                reg_loss += torch.sum(param ** 2)
        reg_loss = (self.lambda_reg / 2) * reg_loss
        
        # ========== Total Loss ==========
        total_loss = radius_loss + hinge_loss + reg_loss
        
        # 监控用：平均距离
        mean_distance = torch.sqrt(distances_squared.mean() + 1e-8)
        
        return {
            'loss': total_loss,
            'radius': radius,
            'radius_squared': radius_squared,
            'radius_loss': radius_loss,
            'hinge_loss': hinge_loss,
            'reg_loss': reg_loss,
            'slack_ratio': slack_ratio,
            'mean_distance': mean_distance
        }
    
    def compute_frr(
        self,
        distances_squared: torch.Tensor,
        radius_squared: torch.Tensor
    ) -> float:
        """
        计算 False Rejection Rate (FRR)
        
        FRR = (dist² > R² 的 Known 样本数) / n
        
        理论上 FRR ≈ ν（±2% 可接受）
        """
        n = distances_squared.size(0)
        rejected = (distances_squared > radius_squared).sum().item()
        return rejected / n


if __name__ == "__main__":
    # 单元测试
    logging.basicConfig(level=logging.INFO)
    
    # 创建虚拟模型
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10, bias=False)
    
    model = DummyModel()
    
    # 创建 loss 函数
    criterion = SVDDLoss(nu=0.1, lambda_reg=1e-6)
    
    # 模拟距离数据 (100 个样本)
    distances_squared = torch.randn(100).abs()  # 确保非负
    
    # 计算 loss
    loss_dict = criterion(distances_squared, model)
    
    print("\n✅ Loss 计算成功:")
    print(f"  Total Loss: {loss_dict['loss']:.6f}")
    print(f"  Radius R: {loss_dict['radius']:.4f}")
    print(f"  R² Loss: {loss_dict['radius_loss']:.6f}")
    print(f"  Hinge Loss: {loss_dict['hinge_loss']:.6f}")
    print(f"  Reg Loss: {loss_dict['reg_loss']:.6f}")
    print(f"  Slack Ratio: {loss_dict['slack_ratio']:.2%} (期望 ≈ {criterion.nu:.0%})")
    print(f"  Mean Distance: {loss_dict['mean_distance']:.4f}")
    
    # 计算 FRR
    frr = criterion.compute_frr(distances_squared, loss_dict['radius_squared'])
    print(f"\n✅ FRR: {frr:.2%} (期望 ≈ {criterion.nu:.0%})")
