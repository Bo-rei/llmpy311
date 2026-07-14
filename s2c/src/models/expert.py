import torch
import torch.nn.functional as F

from src.models.components import ProjectionHead
from src.models.transformer import LoRACausalEncoder


class SmolLMExpert(LoRACausalEncoder):
    def __init__(
        self,
        model_path: str,
        projection_dim: int = 128,
        lora_r: int = 8,
        lora_alpha: int = 32,
        projection_bias: bool = False,
    ):
        super().__init__()
        hidden = self._load_base(
            model_path, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0
        )
        self.projection = ProjectionHead(
            hidden, projection_dim=projection_dim, bias=projection_bias
        )

    def forward(self, input_ids, attention_mask):
        proj = self.projection(self._pooled_features(input_ids, attention_mask))
        features = F.normalize(proj, dim=1)
        return features
