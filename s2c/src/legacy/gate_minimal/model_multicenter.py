"""
Multi-center Deep SVDD Model
============================
可训练多中心 SVDD，用于与 K-means baseline 的因果对照实验。

Architecture:
- K trainable centers (init from K-means)
- Shared R (from quantile)
- Loss: min-distance variant of soft-boundary SVDD

核心修改：
- forward() 返回到最近中心的距离（替代单中心）
- centers 作为可训练参数（替代 frozen buffer）
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class MultiCenterSVDD(nn.Module):
    """
    Multi-center Deep SVDD Gate (仅中心可训练，无 NN projection)
    
    与单中心 SVDD 的唯一区别：
    - 单中心：distance = ||φ(x) - c||
    - 多中心：distance = min_k ||φ(x) - c_k||
    """
    
    def __init__(self, embedding_dim: int, num_centers: int):
        """
        Args:
            embedding_dim: Sentence embedding 维度（384 for all-MiniLM-L6-v2）
            num_centers: 中心数量 K
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_centers = num_centers
        
        # K 个可训练中心 (K, D)
        self.centers = nn.Parameter(torch.zeros(num_centers, embedding_dim))
        
        # 标记：centers 是否已初始化
        self.register_buffer('centers_initialized', torch.tensor(False))
    
    def init_centers_from_kmeans(self, kmeans_centers: torch.Tensor):
        """
        从 K-means 结果初始化 centers（确保与 baseline 公平对比）
        
        Args:
            kmeans_centers: (K, D) tensor from sklearn KMeans
        """
        assert kmeans_centers.shape == (self.num_centers, self.embedding_dim), \
            f"Expected shape ({self.num_centers}, {self.embedding_dim}), got {kmeans_centers.shape}"
        
        with torch.no_grad():
            self.centers.copy_(kmeans_centers)
            self.centers_initialized.fill_(True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算样本到最近中心的距离
        
        Args:
            x: (batch_size, embedding_dim) - frozen sentence embeddings
        
        Returns:
            min_distances: (batch_size,) - 到最近中心的欧氏距离
        """
        # x: (N, D), centers: (K, D)
        # distances: (N, K)
        distances = torch.cdist(x, self.centers, p=2)  # (N, K)
        
        # 取最小距离
        min_distances = distances.min(dim=1)[0]  # (N,)
        
        return min_distances
    
    def get_centers(self) -> torch.Tensor:
        """返回当前 centers（用于分析和可视化）"""
        return self.centers.detach().clone()


class MultiCenterSVDDGate(nn.Module):
    """
    Multi-center Deep SVDD Gate

    约束：
    - MiniLM embeddings 为输入（冻结）
    - 可选投影网络 φ(x)，无 bias，无 BN/LN
    - 多中心可学习参数
    """

    def __init__(
        self,
        input_dim: int = 384,
        proj_dim: int = 64,
        num_centers: int = 16,
        use_mlp: bool = True,
        activation: str = "relu",
        leaky_slope: float = 0.1
    ):
        super().__init__()

        if activation not in {"relu", "leaky_relu"}:
            raise ValueError(f"不支持的激活函数: {activation}")

        act_fn = nn.ReLU() if activation == "relu" else nn.LeakyReLU(negative_slope=leaky_slope)

        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.num_centers = num_centers
        self.use_mlp = use_mlp

        if use_mlp:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, 256, bias=False),
                act_fn,
                nn.Linear(256, proj_dim, bias=False)
            )
        else:
            if input_dim != proj_dim:
                self.proj = nn.Linear(input_dim, proj_dim, bias=False)
            else:
                self.proj = nn.Identity()

        self.centers = nn.Parameter(torch.randn(num_centers, proj_dim))

        logger.info(
            "初始化 MultiCenterSVDDGate: input=%d, proj=%d, K=%d, use_mlp=%s",
            input_dim, proj_dim, num_centers, use_mlp
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, input_dim) MiniLM embeddings

        Returns:
            dict with keys: z, dist_sq, min_dist_sq, assignments
        """
        z = self.proj(x)  # (B, D)
        dist_sq = torch.cdist(z, self.centers, p=2) ** 2  # (B, K)
        min_dist_sq, min_idx = dist_sq.min(dim=1)
        return {
            "z": z,
            "dist_sq": dist_sq,
            "min_dist_sq": min_dist_sq,
            "assignments": min_idx
        }
