"""Tests for the SFT cross-entropy loss."""

import torch
import torch.nn.functional as F


class TestSFTProperties:
    def test_uniform_logits_loss_is_log_vocab(self):
        # Cross-entropy of uniform predictions over V classes is log(V).
        vocab = 50
        logits = torch.zeros(1, 4, vocab)
        labels = torch.randint(0, vocab, (1, 3))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), labels.reshape(-1))
        assert torch.isclose(loss, torch.log(torch.tensor(float(vocab))), atol=1e-4)


# TODO: test nanotrain.losses.sft_cross_entropy once implemented
#   - perfect prediction -> loss ~ 0
#   - response_mask ignores prompt tokens
