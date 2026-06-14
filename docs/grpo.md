# Group Relative Policy Optimization (GRPO)

**Paper:** [DeepSeekMath](https://arxiv.org/abs/2402.03300) (Shao et al. 2024)

## What is GRPO?

GRPO is PPO with the value network removed. For each prompt you sample a *group*
of G completions and use the group's own mean/std as the baseline — no learned
critic needed.

## The Math

```
advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)
L_GRPO      = -(advantage * logprob_response) + beta * KL(policy || ref)
```

Completions above the group average get reinforced, below-average ones get
suppressed. The KL term keeps the policy near the reference.

## Code

- `nanotrain/losses.py` — `grpo_advantages`, `grpo_loss`
- `nanotrain/logprobs.py` — `sequence_log_probs`
- `nanotrain/tiny_models.py` — `TinyGPT.generate` (sampling the group)
- `recipes/grpo.py` — runnable demo

## What to experiment with

- TODO: group size G
- TODO: KL coefficient `beta`
- TODO: mean reward rising over steps
