"""
Tiny models with zero HuggingFace dependency.

Small enough to train in a unit test in milliseconds, but real: a causal
self-attention GPT and a reward model that adds a scalar head on top. They mirror
the HF API just enough for the shared helpers to work — forward(input_ids)
returns an object with a .logits attribute (and .last_hidden_state).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CausalLMOutput:
    """Mirrors the bits of the HF output object the helpers actually read."""
    logits: torch.Tensor
    last_hidden_state: torch.Tensor = None


class TinyGPT(nn.Module):
    """Minimal causal-LM: token+pos embeddings -> N transformer blocks -> head."""

    def __init__(self, vocab_size=32, hidden=32, n_heads=2, n_layers=2, max_len=64):
        super().__init__()
        # TODO: embeddings, blocks (causal self-attention + MLP), final norm, lm_head.
        raise NotImplementedError

    def forward(self, input_ids, attention_mask=None):
        # TODO: return CausalLMOutput(logits=..., last_hidden_state=...)
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=8, temperature=1.0, do_sample=True):
        # TODO: autoregressive sampling loop, used by the GRPO demo.
        raise NotImplementedError


class TinyRewardModel(nn.Module):
    """A backbone plus a scalar reward head read off the last token."""

    def __init__(self, backbone=None, **kwargs):
        super().__init__()
        # TODO: keep/create backbone, add nn.Linear(hidden, 1) reward head.
        raise NotImplementedError

    def forward(self, input_ids, attention_mask=None):
        # TODO: backbone hidden states -> reward at last token -> [B]
        raise NotImplementedError
