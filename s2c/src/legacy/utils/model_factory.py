from transformers import AutoTokenizer
from peft import PeftModel
from typing import Tuple

from legacy.models.expert import SmolLMExpert
from legacy.router import QwenRouter


def load_router(model_path: str, num_classes: int = 10, device: str = 'cpu', lora_r: int = 64, lora_alpha: int = 128) -> Tuple[QwenRouter, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = QwenRouter(model_path, num_classes=num_classes, lora_r=lora_r, lora_alpha=lora_alpha)
    model = model.to(device)
    return model, tokenizer


def load_expert_base(model_path: str, device: str = 'cpu', lora_r: int = 8, lora_alpha: int = 32) -> Tuple[SmolLMExpert, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = SmolLMExpert(model_path, lora_r=lora_r, lora_alpha=lora_alpha)
    model = model.to(device)
    return model, tokenizer


def load_expert_adapter(base_model, adapter_path: str, device: str = 'cpu'):
    """Attach a LoRA adapter to a base model (PeftModel.from_pretrained or load state_dict)
    Returns the base_model with adapter loaded and ready to use."""
    # If base_model is Peft-ready, use PeftModel.from_pretrained
    try:
        base_model = PeftModel.from_pretrained(base_model, adapter_path, device_map=device)
    except Exception:
        # fallback: load adapter weights into base model state_dict
        from pathlib import Path
        import torch
        p = Path(adapter_path)
        if p.is_file():
            state = torch.load(str(p), map_location='cpu')
            base_model.load_state_dict(state, strict=False)
    return base_model
