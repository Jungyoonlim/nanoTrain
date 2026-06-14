"""
Response-only sequence log-probability helpers.

DPO and GRPO both need "how likely is this response under the model?" — a sum
of per-token log-probs over the *response* tokens only (never the prompt). These
helpers compute that for any model whose forward returns an object with .logits
(both TinyGPT and HuggingFace causal LMs qualify).
"""

import torch
import torch.nn.functional as F


def token_log_probs(logits, input_ids):
    """
    Per-token log-prob of the realized next token.

    logits    [B, T, V]
    input_ids [B, T]
    returns   [B, T-1]   token_log_probs[b, t] = log P(input_ids[b, t+1] | <=t)
    """
    # TODO: log_softmax, shift, gather the realized next-token logprob.
    raise NotImplementedError


def sequence_log_probs(model, input_ids, response_mask=None, attention_mask=None, reduction="sum"):
    """
    Run the model and reduce per-token log-probs to per-sequence.

    response_mask [B, T] (optional) restrict to response tokens
    reduction     "sum" -> [B], "mean" -> [B], "none" -> [B, T-1]
    """
    # TODO: forward pass -> token_log_probs -> mask -> reduce.
    raise NotImplementedError
