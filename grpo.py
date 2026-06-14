"""
Group Relative Policy Optimization (GRPO) demo.
Paper: Shao et al., DeepSeekMath (2024) https://arxiv.org/abs/2402.03300

GRPO is PPO without a value network. For each prompt, sample a *group* of G
completions, score them, and use the group mean/std as the baseline:

    advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)
    L_GRPO      = -(advantage * logprob_response) + beta * KL(policy || ref)

Positive-advantage completions get reinforced, negative ones suppressed; KL keeps
the policy near the reference.

=== Usage ===
python grpo.py
"""

import copy

import torch

from nanotrain.losses import grpo_advantages, grpo_loss
from nanotrain.logprobs import sequence_log_probs
from nanotrain.tiny_models import TinyGPT


def reward_fn(sequences):
    """TODO: programmatic reward over sampled token sequences -> [B]."""
    raise NotImplementedError


def train_grpo(steps=200, group_size=8, lr=1e-3, beta=0.04, device="cpu"):
    """
    TODO:
      - policy + frozen ref copies
      - per step: sample group_size completions per prompt (policy.generate)
      - reward_fn -> grpo_advantages -> grpo_loss -> backward -> step
      - log mean reward rising over time
    """
    raise NotImplementedError


if __name__ == "__main__":
    train_grpo()
