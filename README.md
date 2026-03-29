# nanoTrain

A minimal, from-scratch implementation of the modern post-training stack: SFT, reward modeling, DPO, and GRPO in readable PyTorch.

---

**nanoGPT teaches you pretraining, this shows you everything after!**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-11%2F11-brightgreen.svg)]()

## Why nanoTrain?

Most post-training code is **production-focused**. There's abstractions everywhere, impossible to follow the actual algorithm. nanoTrain is **education-focused**, so you can read each file top to bottom and understand exactly what's happening.

Every file maps directly to a paper. Every loss function includes the equation it implements.

For **production**, use [TRL](https://github.com/huggingface/trl), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for multi-GPU, FSDP, 100B+ params. Read nanoTrain *before* those to understand what they're doing.

## Quick Start

```bash
git clone <your-repo>
cd nanoTrain
uv sync

# Run SFT on a tiny built-in dataset (no downloads, ~30 seconds)
python sft.py --dataset tiny --max_steps 30 --batch_size 2
```

## Install

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -e .
```

Requires Python 3.10+ and PyTorch 2.0+.

## The Post-Training Pipeline

| Step | File | Paper | What it does |
|------|------|-------|--------------|
| 1. SFT | [`sft.py`](sft.py) | [InstructGPT (2022)](https://arxiv.org/abs/2203.02155) | Teach the model to follow instructions |
| 2. Reward Model | [`reward.py`](reward.py) | [InstructGPT (2022)](https://arxiv.org/abs/2203.02155) | Learn a scoring function for response quality |
| 3. DPO | [`dpo.py`](dpo.py) | [Rafailov et al. (2023)](https://arxiv.org/abs/2305.18290) | Align using preference pairs, no reward model needed |
| 4. GRPO | [`grpo.py`](grpo.py) | [Shao et al. (2024)](https://arxiv.org/abs/2402.03300) | RL-style alignment with group-relative scoring |

## Experiment Results

### SFT — GPT-2 (124M) on tiny dataset (30 steps)

```
step 0,  loss: 12.3194   <- random guessing across ~50k token vocab
step 10, loss: 0.2161    <- learned the patterns
step 20, loss: 0.1658    <- converging
step 30, loss: 0.1738    <- converged
```

The loss starts at ~12.3 (the model assigns roughly equal probability across its entire 50k vocabulary) and drops to ~0.17 (the model confidently predicts the correct next token). On a tiny dataset this happens fast — with real data, expect slower convergence but better generalization.

### SFT Usage

```bash
# Quick test (built-in, no download)
python sft.py --dataset tiny --max_steps 30 --batch_size 2

# Full training on Alpaca (52k examples)
python sft.py --dataset alpaca --epochs 1 --device cuda

# LIMA (1k curated examples, requires HF auth)
python sft.py --dataset lima --epochs 3 --device cuda

# Custom hyperparameters
python sft.py --dataset alpaca --lr 1e-5 --batch_size 8 --max_steps 500
```

See [`docs/sft.md`](docs/sft.md) for detailed explanation of the math and what to experiment with.

## File Structure

```
sft.py           # Supervised fine-tuning
reward.py        # Bradley-Terry reward model
dpo.py           # Direct Preference Optimization
grpo.py          # Group Relative Policy Optimization
compare.py       # Compare outputs across training stages
tests/           # Mathematical property tests
docs/            # Detailed guides for each algorithm
checkpoints/     # Saved model weights (created during training)
```

## Tests

```bash
pytest tests/
```

Tests verify mathematical properties of each algorithm (e.g., DPO loss symmetry, reward model probabilities, GRPO advantage normalization) — not implementation details.

## License

MIT
