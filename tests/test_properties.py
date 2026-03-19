"""
tests/test_properties.py — Mathematical property tests for post-training algorithms.
"""

import torch
import torch.nn.functional as F

class TestDPOProperties:

    def test_loss_at_reference_equals_log_half(self):
        beta = 0.1
        logp_ratio_w = torch.tensor(0.0)
        logp_ratio_l = torch.tensor(0.0)
        loss = -F.logsigmoid(beta * (logp_ratio_w - logp_ratio_l))
        assert torch.isclose(loss, torch.tensor(0.6931), atol=1e-3)

    def test_loss_decreases_when_preferred_increases(self):
        beta = 0.1
        logp_ratio_w = torch.tensor(1.0)
        logp_ratio_l = torch.tensor(0.0)
        loss_better = -F.logsigmoid(beta * (logp_ratio_w - logp_ratio_l))
        loss_neutral = -F.logsigmoid(beta * torch.tensor(0.0))
        assert loss_better < loss_neutral

    def test_loss_increases_when_rejected_increases(self):
        beta = 0.1
        logp_ratio_w = torch.tensor(0.0)
        logp_ratio_l = torch.tensor(1.0)
        loss_worse = -F.logsigmoid(beta * (logp_ratio_w - logp_ratio_l))
        loss_neutral = -F.logsigmoid(beta * torch.tensor(0.0))
        assert loss_worse > loss_neutral

    def test_beta_controls_sensitivity(self):
        logp_ratio_w = torch.tensor(1.0)
        logp_ratio_l = torch.tensor(-1.0)
        margin = logp_ratio_w - logp_ratio_l
        loss_low_beta = -F.logsigmoid(0.01 * margin)
        loss_high_beta = -F.logsigmoid(1.0 * margin)
        assert loss_high_beta < loss_low_beta

class TestRewardModelProperties:

    def test_equal_rewards_give_fifty_fifty(self):
        r_chosen = torch.tensor(1.0)
        r_rejected = torch.tensor(1.0)
        prob = torch.sigmoid(r_chosen - r_rejected)
        assert torch.isclose(prob, torch.tensor(0.5), atol=1e-5)

    def test_higher_reward_wins(self):
        r_chosen = torch.tensor(2.0)
        r_rejected = torch.tensor(1.0)
        prob = torch.sigmoid(r_chosen - r_rejected)
        assert prob > 0.5

    def test_preference_loss_is_cross_entropy(self):
        r_chosen = torch.tensor(2.0)
        r_rejected = torch.tensor(0.5)
        loss_direct = -F.logsigmoid(r_chosen - r_rejected)
        prob = torch.sigmoid(r_chosen - r_rejected)
        loss_bce = F.binary_cross_entropy(prob, torch.tensor(1.0))
        assert torch.isclose(loss_direct, loss_bce, atol=1e-5)

class TestGRPOProperties:

    def test_advantages_sum_to_zero(self):
        rewards = torch.tensor([1.0, 3.0, 2.0, 5.0])
        mean = rewards.mean()
        std = rewards.std()
        advantages = (rewards - mean) / (std + 1e-8)
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
