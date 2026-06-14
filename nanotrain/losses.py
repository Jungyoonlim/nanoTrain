"""
Pure post-training losses.

Every function here takes plain tensors and returns a tensor. No model calls,
no data loading, no optimizers — just the math. That keeps each algorithm easy
to unit-test and easy to read next to the equation it implements.
"""

import torch
import torch.nn.functional as F


def sft_cross_entropy(logits, input_ids, response_mask=None):
    """
    SFT loss = -1/T * Σ log P(token_t | token_<t)   (next-token cross-entropy).

    logits        [B, T, V]
    input_ids     [B, T]
    response_mask [B, T] (optional) 1 for tokens to train on, 0 to ignore
                  (e.g. mask out the prompt so only the response is learned)
    """
    # TODO: shift logits/labels by one, cross-entropy, apply response_mask.
    raise NotImplementedError


def reward_bt_loss(chosen_rewards, rejected_rewards):
    """
    Bradley-Terry preference loss = -log σ(r_chosen - r_rejected).

    chosen_rewards, rejected_rewards: [B] scalar rewards per sequence.
    """
    # TODO
    raise NotImplementedError


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
):
    """
    DPO loss = -log σ( beta * (policy_logratio - ref_logratio) )

      policy_logratio = logp_policy(chosen) - logp_policy(rejected)
      ref_logratio    = logp_ref(chosen)    - logp_ref(rejected)

    All four inputs are [B] summed sequence log-probs. Returns (loss, metrics).
    """
    # TODO
    raise NotImplementedError


def grpo_advantages(rewards, eps=1e-8):
    """
    Group-relative advantages = (rewards - mean) / (std + eps).

    rewards: [G] rewards for the G sampled completions of one prompt.
    """
    # TODO
    raise NotImplementedError


def grpo_loss(token_logprobs, advantages, response_mask=None, ref_logprobs=None, beta=0.0):
    """
    GRPO policy-gradient loss: -(advantage * logprob), averaged over response
    tokens, plus an optional beta * KL(policy || ref) penalty.

    token_logprobs [B, T] per-token log-probs under the current policy
    advantages     [B]    one group-relative advantage per sequence
    """
    # TODO
    raise NotImplementedError
