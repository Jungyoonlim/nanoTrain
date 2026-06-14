# Direct Preference Optimization (DPO)

**Paper:** [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (Rafailov et al. 2023)

## What is DPO?

DPO aligns a model from preference pairs *without* training a separate reward
model or running PPO. It optimizes the policy's log-probabilities directly so it
prefers chosen over rejected — more strongly than a frozen reference model does.

## The Math

```
policy_logratio = logp_policy(chosen) - logp_policy(rejected)
ref_logratio    = logp_ref(chosen)    - logp_ref(rejected)
L_DPO           = -log σ( beta * (policy_logratio - ref_logratio) )
```

`beta` controls how far the policy may drift from the reference. The reference is
the SFT model, frozen.

## Code

- `nanotrain/losses.py` — `dpo_loss`
- `nanotrain/logprobs.py` — `sequence_log_probs` (response-only)
- `recipes/dpo.py` — runnable demo

## What to experiment with

- TODO: sweep `beta`
- TODO: chosen/rejected reward margin over training
