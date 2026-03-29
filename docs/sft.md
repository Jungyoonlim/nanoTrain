# Supervised Fine-Tuning (SFT)

**Paper:** [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT, Ouyang et al. 2022)

## What is SFT?

SFT is the first step of post-training. You take a pretrained model (which only knows how to complete text) and teach it to follow instructions by training on high-quality instruction/response pairs.

**The intuition:** Given a good response, make the model more likely to produce those exact tokens.

## The Math

```
Loss = -1/T * Σ log P(token_t | token_<t)
```

This is just **cross-entropy loss** — for each token in the response, how surprised was the model? Lower loss = less surprise = model is learning.

- `P(token_t | token_<t)` — probability the model assigns to the correct next token, given everything before it
- The `log` makes this a log-probability (easier to work with numerically)
- The negative sign flips it so we're **minimizing** loss (maximizing probability)
- `1/T` averages over all `T` tokens

## Usage

```bash
# Quick test with built-in tiny dataset (no downloads needed)
python sft.py --dataset tiny --max_steps 30 --batch_size 2

# Train on Stanford Alpaca (~52k instruction/response pairs)
python sft.py --dataset alpaca --epochs 1

# Train on LIMA (1k high-quality examples, requires HF auth)
python sft.py --dataset lima --epochs 3

# Customize training
python sft.py --dataset alpaca --lr 1e-5 --batch_size 8 --max_steps 500 --device cuda
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `gpt2` | HuggingFace model name (e.g. `gpt2`, `gpt2-medium`) |
| `--dataset` | `lima` | Dataset: `tiny`, `alpaca`, or `lima` |
| `--epochs` | `1` | Number of passes through the dataset |
| `--batch_size` | `4` | Examples per gradient update |
| `--lr` | `2e-5` | Learning rate |
| `--max_steps` | `None` | Stop after N steps (useful for quick tests) |
| `--device` | `cpu` | `cpu` or `cuda` |

## Example Results

Training GPT-2 (124M params) on the tiny dataset for 30 steps:

```
step 0,  loss: 12.3194   <- random guessing across ~50k vocab
step 10, loss: 0.2161    <- learned the patterns
step 20, loss: 0.1658    <- converging
step 30, loss: 0.1738    <- converged
```

The model checkpoint is saved to `checkpoints/sft/`.

## What to Experiment With

- **Learning rate:** Try `1e-4` (faster, less stable) vs `1e-6` (slower, more stable). Watch if loss spikes — that means lr is too high.
- **Dataset size:** Compare tiny (100 examples, memorizes) vs alpaca (52k, generalizes). Small datasets overfit quickly.
- **Epochs:** More epochs on small datasets = more overfitting. On large datasets, 1-3 epochs is typical.
- **Model size:** Try `gpt2` (124M) vs `gpt2-medium` (355M) — bigger models learn faster but use more memory.

## What Comes Next?

SFT teaches the model the *format* of good responses, but it doesn't teach it to *prefer* good responses over bad ones. That's what the rest of the pipeline does:

1. **Reward Model** (`reward.py`) — learn a scoring function for response quality
2. **DPO** (`dpo.py`) — align the model using preference pairs, no reward model needed
3. **GRPO** (`grpo.py`) — RL-style alignment using group-relative scoring
