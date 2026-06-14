# Reward Modeling

**Paper:** [InstructGPT](https://arxiv.org/abs/2203.02155) (Ouyang et al. 2022)

## What is a reward model?

A reward model scores a response with a single scalar: how good is it? You train
it from human preference pairs — for a prompt, a *chosen* response and a
*rejected* one — and it learns to give the chosen one a higher score.

## The Math

Bradley-Terry model of pairwise preference:

```
P(y_w > y_l | x) = σ(r(y_w) - r(y_l))
Loss             = -log σ(r(y_w) - r(y_l))
```

Only the *difference* in rewards matters, so the absolute scale is arbitrary.

## Code

- `nanotrain/losses.py` — `reward_bt_loss`
- `nanotrain/tiny_models.py` — `TinyRewardModel` (backbone + scalar head)
- `reward.py` — runnable demo

## What to experiment with

- TODO: chosen/rejected margin over training
- TODO: where the reward is read off (last token vs. mean-pooled)
