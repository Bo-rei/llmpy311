#!/usr/bin/env python3
"""
TASK-201: Backbone Module with LoRA Support

Provides generative and representation-learning forward modes for SmolLM2-1.7B.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)


class SmolLM2Backbone(nn.Module):
    """SmolLM2-1.7B backbone with LoRA support"""
    
    def __init__(self, model_path, device='cuda', apply_lora=False, lora_config=None):
        super().__init__()
        
        self.device = device
        self.model_path = model_path
        
        # Load model and tokenizer
        logger.info(f"Loading SmolLM2-1.7B from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            local_files_only=True,
            device_map=device if isinstance(device, str) else None,
            torch_dtype=torch.float32
        )
        
        if device != 'cpu':
            self.model = self.model.to(device)
        
        self.model.eval()
        
        # Apply LoRA if requested
        if apply_lora and lora_config:
            self._apply_lora(lora_config)
        
        self.hidden_size = self.model.config.hidden_size
        logger.info(f"Backbone initialized. Hidden size: {self.hidden_size}")
    
    def _apply_lora(self, lora_config):
        """Apply LoRA to attention layers"""
        try:
            from peft import get_peft_model, LoraConfig
            
            peft_config = LoraConfig(
                r=lora_config.get('r', 8),
                lora_alpha=lora_config.get('lora_alpha', 16),
                target_modules=lora_config.get('target_modules', ['q_proj', 'v_proj']),
                lora_dropout=lora_config.get('lora_dropout', 0.05),
                bias='none',
                task_type='CAUSAL_LM'
            )
            self.model = get_peft_model(self.model, peft_config)
            logger.info(f"LoRA applied. r={peft_config.r}")
        except ImportError:
            logger.warning("peft not installed. Skipping LoRA.")
    
    def forward_generative(self, input_ids, attention_mask=None):
        """Forward pass for generative (Causal LM) training"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        return outputs
    
    def forward_representation(self, input_ids, attention_mask=None):
        """Forward pass to extract representation (last hidden state, mean pooled)"""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Mean pooling
        last_hidden = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            masked_output = last_hidden * mask
            sum_output = masked_output.sum(dim=1)
            sum_mask = mask.sum(dim=1)
            pooled = sum_output / sum_mask
        else:
            pooled = last_hidden.mean(dim=1)
        
        return pooled
    
    def get_hidden_size(self):
        return self.hidden_size
    
    def encode(self, texts, tokenize=True):
        """Encode texts to embeddings"""
        if tokenize:
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = texts
        
        return self.forward_representation(**inputs)


def load_backbone(model_path, device='cuda', apply_lora=False, lora_config=None):
    """Factory function to load backbone"""
    return SmolLM2Backbone(model_path, device, apply_lora, lora_config)
