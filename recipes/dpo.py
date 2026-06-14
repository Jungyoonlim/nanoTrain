"""
Direct Preference Optimization (DPO) demo.
Paper: Rafailov et al. (2023) https://arxiv.org/abs/2305.18290

DPO trains directly on preference pairs (prompt, chosen, rejected). It pushes the
policy to prefer chosen over rejected more strongly than a frozen reference does:

    policy_logratio = logp_policy(chosen) - logp_policy(rejected)
    ref_logratio    = logp_ref(chosen)    - logp_ref(rejected)
    L_DPO           = -log σ( beta * (policy_logratio - ref_logratio) )

No separate reward model, no PPO. beta controls how far the policy may drift from
the reference.

=== Usage ===
python -m recipes.dpo
"""

import copy

import torch

from nanotrain.losses import dpo_loss
from nanotrain.logprobs import sequence_log_probs
from nanotrain.tiny_models import TinyGPT


def make_preference_data():
    """TODO: return chosen/rejected token ids + response masks."""
    raise NotImplementedError


def train_dpo(steps=200, lr=1e-4, beta=0.1, device="cpu"):
    """
    TODO:
      - policy = copy of base model (trainable); ref = frozen copy
      - per step: sequence_log_probs(policy/ref, chosen/rejected) over response only
      - dpo_loss(...) -> backward -> step
      - log the chosen/rejected reward margin
    """
    raise NotImplementedError


if __name__ == "__main__":
    train_dpo()
