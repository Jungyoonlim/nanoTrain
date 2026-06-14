"""Tests for the response-only log-probability helpers."""

import torch
import torch.nn.functional as F


class TestLogProbProperties:
    def test_log_probs_are_non_positive(self):
        # log of a probability is always <= 0.
        logits = torch.randn(2, 5, 8)
        lp = F.log_softmax(logits, dim=-1)
        assert (lp <= 0).all()


# TODO: test nanotrain.logprobs.token_log_probs / sequence_log_probs
#   - token_log_probs returns shape [B, T-1]
#   - response_mask zeroes out prompt tokens before the sum
#   - matches a hand-computed value on a fixed tiny example
