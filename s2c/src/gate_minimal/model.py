"""
Minimal Deep SVDD Gate Model

严格实现 Deep SVDD (ICML 2018) 论文要求：
1. Bias-free MLP (防止常数塌陷)
2. 无 BatchNorm/LayerNorm (保持几何可塑性)
3. 无界激活 ReLU (防止特征有界)
4. Center c 冻结为 buffer (不参与训练)
5. 半径 R 从分位数计算 (不参与梯度)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class MinimalSVDDGate(nn.Module):
    """
    Minimal Deep SVDD Gate (纯 MLP, 无预训练模型)
    
    Architecture (论文标准):
        Input (d_in) 
            → Linear(d_in → 1024, bias=False)
            → ReLU
            → Linear(1024 → 256, bias=False)  
            → ReLU
            → Linear(256 → d_svdd, bias=False)
        
    Constraints (硬性约束):
        - 所有 Linear 必须 bias=False
        - 不允许 BatchNorm/LayerNorm
        - 不允许 Dropout
        - 只允许 ReLU/LeakyReLU (无界激活)
        - Center c 必须是 buffer (不参与梯度)
    
    Reference:
        Eq. (3) in Ruff et al., "Deep One-Class Classification" (ICML 2018)
    """
    
    def __init__(
        self,
        input_dim: int = 384,  # 预计算 embedding 维度 (如 all-MiniLM-L6-v2)
        hidden_dim: int = 1024,
        intermediate_dim: int = 256,
        svdd_dim: int = 64,  # SVDD 空间维度
        activation: str = 'relu',  # 'relu' or 'leaky_relu'
        leaky_slope: float = 0.1,
        center: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        logger.info(f"初始化 MinimalSVDDGate: {input_dim} → {hidden_dim} → {intermediate_dim} → {svdd_dim}")
        
        # 激活函数选择 (论文推荐 ReLU, 也可用 LeakyReLU 防止 dead neuron)
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(negative_slope=leaky_slope)
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # MLP 结构 (严格无 bias)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),  # 硬性约束1
            act_fn,
            nn.Linear(hidden_dim, intermediate_dim, bias=False),  # 硬性约束2
            act_fn,
            nn.Linear(intermediate_dim, svdd_dim, bias=False)  # 硬性约束3
        )
        
        # Xavier 初始化 (论文标准)
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        
        # Center c (固定，不参与训练)
        if center is None:
            # 初始化为零向量（后续必须通过 init_center.py 替换）
            center = torch.zeros(svdd_dim)
            logger.warning("⚠️ Center 初始化为零向量，训练前必须用 Known 数据重新初始化！")
        
        self.register_buffer('center', center)  # 注册为 buffer（非参数）
        logger.info(f"✅ Center 已注册为 buffer: shape={center.shape}")
        
        self.svdd_dim = svdd_dim
        self.input_dim = input_dim
        
    def forward(
        self,
        embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            embeddings: (batch_size, input_dim) - 预计算的文本 embedding
        
        Returns:
            {
                'features': (batch_size, svdd_dim) - SVDD 空间特征 (RAW, 无归一化),
                'distances': (batch_size,) - 到 center 的欧氏距离 ||φ(x) - c||,
                'distances_squared': (batch_size,) - 距离平方 ||φ(x) - c||²
            }
        """
        # MLP 编码 (φ(x; W) in 论文公式)
        features = self.encoder(embeddings)  # (batch, svdd_dim)
        
        # 计算到 center 的欧氏距离 (论文 Eq. 3)
        # dist² = ||φ(x) - c||²
        diff = features - self.center.unsqueeze(0)  # (batch, svdd_dim)
        distances_squared = torch.sum(diff ** 2, dim=1)  # (batch,)
        distances = torch.sqrt(distances_squared + 1e-8)  # 数值稳定性
        
        return {
            'features': features,
            'distances': distances,
            'distances_squared': distances_squared
        }
    
    def update_center(self, new_center: torch.Tensor):
        """
        更新 center (仅在初始化阶段调用一次)
        
        Args:
            new_center: (svdd_dim,) - 从 Known 数据计算的均值
        """
        if new_center.shape != self.center.shape:
            raise ValueError(f"Center 形状不匹配: {new_center.shape} != {self.center.shape}")
        
        self.center.copy_(new_center)
        logger.info(f"✅ Center 已更新: {new_center[:5].tolist()}...")
    
    def get_trainable_params_count(self) -> Dict[str, int]:
        """统计可训练参数数量"""
        total = 0
        trainable = 0
        
        for param in self.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        
        # Center 不应该可训练
        assert not self.center.requires_grad, "❌ Center 不应该参与训练！"
        
        logger.info(f"参数统计: {trainable:,} 可训练 / {total:,} 总计 ({100*trainable/total:.2f}%)")
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


if __name__ == "__main__":
    # 单元测试
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型
    model = MinimalSVDDGate(
        input_dim=384,  # all-MiniLM-L6-v2 embedding
        svdd_dim=64
    )
    
    # 检查参数
    model.get_trainable_params_count()
    
    # 测试 forward
    batch_size = 8
    embeddings = torch.randn(batch_size, 384)
    
    outputs = model(embeddings)
    
    print("\n✅ Forward Pass 成功:")
    print(f"  Features shape: {outputs['features'].shape}")
    print(f"  Distances: {outputs['distances']}")
    print(f"  Distances²: {outputs['distances_squared']}")
    
    # 验证约束
    print("\n✅ 约束验证:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None, f"❌ {name} 不应该有 bias！"
            print(f"  ✓ {name}: bias=None")
    
    print("\n✅ Center 验证:")
    print(f"  Center requires_grad: {model.center.requires_grad} (应为 False)")
    print(f"  Center shape: {model.center.shape}")
