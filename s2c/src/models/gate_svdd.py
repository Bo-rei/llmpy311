"""
HiLSA-MoE v19.2 - Deep SVDD Gate Model

核心约束:
1. Bias-free Linear Layers (防止常数塌陷)
2. Unbounded Activation (LeakyReLU, 防止特征有界)
3. L2 Normalization (特征归一化)
4. Fixed Center C (训练期间冻结)
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict
import logging
from pathlib import Path
from src.runtime import WorkspacePaths

logger = logging.getLogger(__name__)

DEFAULT_BACKBONE_PATH = str(WorkspacePaths.discover(Path(__file__)).smollm17b)


class BiasFreeMLP(nn.Module):
    """无偏置投影头 (Bias-free Projection Head)
    
    约束:
    - 所有 Linear 层必须 bias=False
    - 使用无界激活 LeakyReLU (防止特征塌陷到有界空间)
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 128,
        negative_slope: float = 0.1
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),  # 硬性约束1
            nn.LeakyReLU(negative_slope=negative_slope),    # 硬性约束2: 无界激活
            nn.Linear(hidden_dim, output_dim, bias=False)  # 硬性约束3
        )
        
        # 初始化（避免梯度消失）
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """
        return self.projection(x)


class SVDDGate(nn.Module):
    """Deep SVDD Gate (全局门控)
    
    架构:
    - Backbone: SmolLM2-1.7B (Frozen)
    - Adapter: LoRA (r=64, alpha=128)
    - Projection Head: Bias-free MLP
    - Center: Fixed (初始化后冻结)
    """
    
    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE_PATH,
        lora_r: int = 64,
        lora_alpha: int = 128,
        projection_dim: int = 128,
        center: Optional[torch.Tensor] = None,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        logger.info(f"初始化 SVDD Gate: {backbone_name}")
        
        # 加载 Backbone (Frozen) - 使用 bfloat16 提升性能
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16  # RTX 5070 支持 bfloat16
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone已冻结")
        
        # 配置 LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],  # 保持与checkpoint一致
            bias="none"  # 严格无偏置
        )
        
        self.backbone = get_peft_model(self.backbone, lora_config)
        logger.info(f"LoRA已应用: r={lora_r}, alpha={lora_alpha}")
        
        # Projection Head (Bias-free)
        hidden_size = self.backbone.config.hidden_size
        self.projection_head = BiasFreeMLP(
            input_dim=hidden_size,
            hidden_dim=512,
            output_dim=projection_dim
        )
        
        # 中心点 C (固定)
        if center is None:
            # 初始化为零向量（后续需要通过 init_center.py 替换）
            center = torch.zeros(projection_dim)
        
        self.register_buffer('center', center)  # 注册为buffer（不参与梯度）
        logger.info(f"Center已注册: shape={center.shape}")
        
        self.projection_dim = projection_dim
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            {
                'features': (batch_size, projection_dim) - L2归一化后的特征,
                'distances': (batch_size,) - 到中心的欧氏距离,
                'raw_features': (batch_size, projection_dim) - 未归一化特征
            }
        """
        # Backbone Forward (只需 last_hidden_state)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 取最后一层的 [CLS] token 或 Mean Pooling
        # SmolLM2 使用 last_hidden_state 的 mean pooling
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        # Mean pooling (考虑 attention_mask)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask  # (batch, hidden_size)
        
        # Projection (Bias-free) - RAW EUCLIDEAN SPACE
        projected = self.projection_head(pooled)  # (batch, projection_dim)
        features = projected  # NO NORMALIZATION - raw embeddings
        
        # 计算到中心的距离 (Euclidean L2)
        distances = torch.norm(features - self.center.unsqueeze(0), p=2, dim=1)
        
        return {
            'features': features,
            'distances': distances,
            'raw_features': projected
        }
    
    def compute_svdd_loss(
        self,
        features: torch.Tensor,
        nu: float = 0.1,
        radius_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """ν-SVDD Loss (Soft-Boundary Deep SVDD)
        
        Loss = mean(||z - C||) + λ * R
        R = quantile(||z - C||, 1 - ν)
        
        Theory: 
        - Allows ν fraction of Known samples as outliers (slack)
        - R penalty prevents hypersphere from collapsing
        - No variance penalty (maintains generative SVDD nature)
        - OOS detection: distance > R + margin
        
        Args:
            features: (batch_size, projection_dim) - RAW embeddings
            nu: Fraction of outliers allowed (default 0.1 = 10%)
            radius_weight: Weight for R penalty (default 1.0)
        
        Returns:
            {
                'loss': total loss,
                'svdd_loss': mean distance to center,
                'radius': soft boundary radius R,
                'radius_loss': R penalty term,
                'reg_loss': 0.0 (compatibility)
            }
        """
        # Compute Euclidean distances to center
        distances = torch.norm(features - self.center.unsqueeze(0), p=2, dim=1)
        
        # SVDD Loss: mean distance
        svdd_loss = torch.mean(distances)
        
        # Soft Boundary Radius R: (1-ν) quantile
        # e.g., ν=0.1 → take 90th percentile as radius
        quantile_value = 1.0 - nu
        radius = torch.quantile(distances, quantile_value)
        
        # Radius penalty: prevents R from shrinking to zero
        radius_loss = radius
        
        # Total loss
        total_loss = svdd_loss + radius_weight * radius_loss
        
        return {
            'loss': total_loss,
            'svdd_loss': svdd_loss,
            'radius': radius,
            'radius_loss': radius_loss,
            'reg_loss': torch.tensor(0.0, device=features.device)
        }
    
    def update_center(self, new_center: torch.Tensor):
        """更新中心 (仅在初始化时调用一次)"""
        self.center.copy_(new_center)
        logger.info(f"Center已更新: {new_center[:5]}...")
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float
    ) -> Dict[str, torch.Tensor]:
        """推理: 判断是否为 OOS
        
        Args:
            threshold: 校准后的半径 R
        
        Returns:
            {
                'distances': (batch_size,),
                'predictions': (batch_size,) - 0=Known, 1=OOS
            }
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            distances = outputs['distances']
            predictions = (distances > threshold).long()
        
        return {
            'distances': distances,
            'predictions': predictions
        }
    
    def get_trainable_params(self):
        """获取可训练参数（仅 LoRA + Projection Head）"""
        trainable = []
        total = 0
        
        for name, param in self.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable.append((name, param.numel()))
        
        trainable_count = sum(count for _, count in trainable)
        
        logger.info(f"可训练参数: {trainable_count:,} / {total:,} ({100*trainable_count/total:.2f}%)")
        
        return trainable


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型
    model = SVDDGate(
        backbone_name=DEFAULT_BACKBONE_PATH,
        lora_r=64,
        lora_alpha=128,
        projection_dim=128
    )
    
    # 打印可训练参数
    model.get_trainable_params()
    
    # 测试前向传播
    batch_size = 4
    seq_len = 32
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    outputs = model(input_ids, attention_mask)
    
    print(f"\nFeatures shape: {outputs['features'].shape}")
    print(f"Distances shape: {outputs['distances'].shape}")
    print(f"Distances: {outputs['distances']}")
    
    # 测试损失计算
    loss_dict = model.compute_svdd_loss(outputs['features'])
    print(f"\nSVDD Loss: {loss_dict['svdd_loss']:.4f}")
    print(f"Reg Loss: {loss_dict['reg_loss']:.4f}")
    print(f"Total Loss: {loss_dict['loss']:.4f}")
