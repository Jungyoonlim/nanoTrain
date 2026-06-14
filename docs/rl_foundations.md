# RL Foundations

Background shared by DPO and GRPO. The goal: read this once, then the two
algorithm docs make sense.

## The objective

RLHF maximizes expected reward while staying close to the reference (SFT) model:

```
max_π  E[r(x, y)]  -  beta * KL(π || π_ref)
```

The KL term is the leash — without it the policy collapses onto whatever the
reward model loves, including degenerate text.

## Sequence log-probabilities

- TODO: why we sum log-probs over *response* tokens only (never the prompt)
- TODO: response masking

## Policy gradient

- TODO: REINFORCE, `∇ log π * advantage`
- TODO: why a baseline reduces variance

## Advantages & baselines

- TODO: value-network baseline (PPO) vs. group baseline (GRPO)

## KL penalty

- TODO: the unbiased k3 KL estimator `exp(r) - r - 1`

## Where each algorithm sits

| | Reward model | Critic | Baseline |
|---|---|---|---|
| DPO | none (implicit) | none | reference model |
| GRPO | explicit / programmatic | none | group mean |
| PPO | explicit | value net | learned value |
