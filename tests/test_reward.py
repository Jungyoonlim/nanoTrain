"""Tests for the Bradley-Terry reward model loss."""

import torch
import torch.nn.functional as F


class TestRewardModelProperties:
    def test_equal_rewards_give_fifty_fifty(self):
        prob = torch.sigmoid(torch.tensor(1.0) - torch.tensor(1.0))
        assert torch.isclose(prob, torch.tensor(0.5), atol=1e-5)

    def test_higher_reward_wins(self):
        prob = torch.sigmoid(torch.tensor(2.0) - torch.tensor(1.0))
        assert prob > 0.5

    def test_preference_loss_is_cross_entropy(self):
        r_chosen, r_rejected = torch.tensor(2.0), torch.tensor(0.5)
        loss_direct = -F.logsigmoid(r_chosen - r_rejected)
        prob = torch.sigmoid(r_chosen - r_rejected)
        loss_bce = F.binary_cross_entropy(prob, torch.tensor(1.0))
        assert torch.isclose(loss_direct, loss_bce, atol=1e-5)


# TODO: test nanotrain.losses.reward_bt_loss + TinyRewardModel training
#   - chosen reward rises above rejected over training steps
