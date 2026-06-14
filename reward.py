"""
Reward modeling demo.
Paper: Ouyang et al., InstructGPT (2022) https://arxiv.org/abs/2203.02155

Bradley-Terry model: P(y_w > y_l | x) = σ(r(y_w) - r(y_l))
Loss:               -log σ(r(y_w) - r(y_l))

Train a scalar reward head so chosen responses score higher than rejected ones.

=== Usage ===
python reward.py
"""

from nanotrain.losses import reward_bt_loss
from nanotrain.tiny_models import TinyRewardModel


def make_preference_data():
    """TODO: return (chosen_ids, rejected_ids) tensors of token sequences."""
    raise NotImplementedError


def train_reward_model(steps=200, lr=1e-3):
    """TODO: forward chosen/rejected -> reward_bt_loss -> step; log accuracy."""
    raise NotImplementedError


if __name__ == "__main__":
    train_reward_model()
