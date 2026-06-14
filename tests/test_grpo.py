"""Tests for GRPO group-relative advantages."""

import torch


class TestGRPOProperties:
    def test_advantages_sum_to_zero(self):
        rewards = torch.tensor([1.0, 3.0, 2.0, 5.0])
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        assert torch.isclose(advantages.sum(), torch.tensor(0.0), atol=1e-5)

    def test_best_response_gets_positive_advantage(self):
        rewards = torch.tensor([1.0, 3.0, 2.0, 5.0])
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        assert advantages[rewards.argmax()] > 0

    def test_worst_response_gets_negative_advantage(self):
        rewards = torch.tensor([1.0, 3.0, 2.0, 5.0])
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        assert advantages[rewards.argmin()] < 0

    def test_uniform_rewards_give_zero_advantages(self):
        rewards = torch.tensor([3.0, 3.0, 3.0, 3.0])
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-5)


# TODO: test nanotrain.losses.grpo_advantages / grpo_loss against the above
