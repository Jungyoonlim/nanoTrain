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


For **Production framework**, use [TRL](https://github.com/huggingface/trl), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) for multi-GPU, FSDP, 100B+ params

**Educational foundation**: You can read nanoTrain *before* those to understand what they're doing


## Quick Start

```bash
# Set up with uv (recommended)
git clone <your-repo>
cd nanotrain
uv sync
```

## What it does

Every file maps directly to a paper. Every loss function includes the equation it implements. Every algorithm includes a plain-English explanation of what it's really doing, what breaks, and what to experiment with.


## Further Reading



## License
 
MIT