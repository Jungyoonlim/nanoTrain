"""Tests for the DPO loss."""

import torch
import torch.nn.functional as F


class TestDPOProperties:
    def test_loss_at_reference_equals_log_half(self):
        beta = 0.1
        loss = -F.logsigmoid(beta * (torch.tensor(0.0) - torch.tensor(0.0)))
        assert torch.isclose(loss, torch.tensor(0.6931), atol=1e-3)

    def test_loss_decreases_when_preferred_increases(self):
        beta = 0.1
        loss_better = -F.logsigmoid(beta * (torch.tensor(1.0) - torch.tensor(0.0)))
        loss_neutral = -F.logsigmoid(beta * torch.tensor(0.0))
        assert loss_better < loss_neutral

    def test_loss_increases_when_rejected_increases(self):
        beta = 0.1
        loss_worse = -F.logsigmoid(beta * (torch.tensor(0.0) - torch.tensor(1.0)))
        loss_neutral = -F.logsigmoid(beta * torch.tensor(0.0))
        assert loss_worse > loss_neutral

    def test_beta_controls_sensitivity(self):
        margin = torch.tensor(1.0) - torch.tensor(-1.0)
        loss_low_beta = -F.logsigmoid(0.01 * margin)
        loss_high_beta = -F.logsigmoid(1.0 * margin)
        assert loss_high_beta < loss_low_beta


# TODO: test nanotrain.losses.dpo_loss
#   - equal policy/ref logratios -> loss = log 2
#   - gradient raises chosen logratio relative to rejected
