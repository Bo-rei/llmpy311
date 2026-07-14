"""LLM uncertainty verifier with deterministic prompt template v1.

Replaces the existing LLMSemanticVerifier with a stricter, versioned
prompt template and clear uncertainty estimation interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

PROMPT_TEMPLATE_VERSION = "v1"


@dataclass
class LLMUncertaintyVerifierConfig:
    """Configuration for LLMUncertaintyVerifier."""

    max_length: int = 128
    yes_token: str = " yes"
    no_token: str = " no"
    prompt_version: str = PROMPT_TEMPLATE_VERSION


class LLMUncertaintyVerifier:
    """Compute ID-probability scores using LLM next-token likelihood.

    Uses a deterministic prompt template (version 1) to compute the conditional
    probability P(id | query, candidate_intent) via Yes/No token likelihoods.

    Args:
        tokenizer: HuggingFace-compatible tokenizer.
        model: SmolLMRouter or any model with .base attribute exposing logits.
        device: Torch device to run inference on.
        config: Verifier configuration.
    """

    def __init__(
        self,
        tokenizer,
        model,
        device: torch.device,
        config: LLMUncertaintyVerifierConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.config = config or LLMUncertaintyVerifierConfig()

        yes_ids = self.tokenizer(self.config.yes_token, add_special_tokens=False)["input_ids"]
        no_ids = self.tokenizer(self.config.no_token, add_special_tokens=False)["input_ids"]
        self.yes_token_id = int(yes_ids[0]) if yes_ids else int(self.tokenizer.eos_token_id)
        self.no_token_id = int(no_ids[0]) if no_ids else int(self.tokenizer.eos_token_id)

    @staticmethod
    def _intent_description(intent_name: str) -> str:
        """Generate a deterministic description from an intent label.

        Args:
            intent_name: Raw intent label string (underscores allowed).

        Returns:
            Human-readable intent description.
        """
        normalized = intent_name.replace("_", " ").strip()
        return f"User intent topic: {normalized}."

    def _build_prompt_v1(self, query: str, intent_name: str) -> str:
        """Build deterministic prompt template v1.

        Template is fixed and must not be modified for reproducibility.

        Args:
            query: User query text.
            intent_name: Candidate intent label.

        Returns:
            Formatted prompt string.
        """
        description = self._intent_description(intent_name)
        return (
            "Task: Decide if the query matches the described intent.\n"
            "Respond with exactly yes or no.\n"
            f"Intent: {intent_name}\n"
            f"Description: {description}\n"
            f"Query: {query}\n"
            "Answer:"
        )

    @torch.no_grad()
    def predict_id_probability(
        self,
        query_text: str,
        candidate_intents: List[str],
    ) -> float:
        """Compute the probability that a query is in-distribution (ID).

        For each candidate intent, compute P(yes | prompt). Return the
        maximum yes-probability across all candidate intents.

        Args:
            query_text: User query text.
            candidate_intents: List of candidate known intent labels.

        Returns:
            Maximum P(yes) score in [0, 1]. Higher means more likely ID.
        """
        if not candidate_intents:
            return 0.0

        max_id_prob = 0.0
        for intent_name in candidate_intents:
            prompt = self._build_prompt_v1(str(query_text), str(intent_name))
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self.model.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            yes_p = float(probs[0, self.yes_token_id].item())
            no_p = float(probs[0, self.no_token_id].item())
            denom = yes_p + no_p
            score = yes_p / denom if denom > 1e-12 else 0.5
            if score > max_id_prob:
                max_id_prob = score

        return float(max_id_prob)

    @torch.no_grad()
    def predict_id_probability_batch(
        self,
        query_texts: Sequence[str],
        top_candidate_intents: Sequence[str],
    ) -> List[float]:
        """Compute ID probabilities for a batch, one candidate per query.

        For fast batched evaluation where each query has exactly one
        candidate intent (e.g., the top prototype match).

        Args:
            query_texts: Batch of user query texts.
            top_candidate_intents: One candidate intent label per query.

        Returns:
            List of P(yes) scores in [0, 1].
        """
        if len(query_texts) != len(top_candidate_intents):
            raise ValueError("query_texts and top_candidate_intents must have equal length.")

        scores: List[float] = []
        for query, intent_name in zip(query_texts, top_candidate_intents):
            prompt = self._build_prompt_v1(str(query), str(intent_name))
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self.model.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            yes_p = float(probs[0, self.yes_token_id].item())
            no_p = float(probs[0, self.no_token_id].item())
            denom = yes_p + no_p
            score = yes_p / denom if denom > 1e-12 else 0.5
            scores.append(float(score))

        return scores
