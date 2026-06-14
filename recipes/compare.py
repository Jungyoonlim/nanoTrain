"""
Compare outputs across training stages.

Ask the same question to checkpoints from different points in the pipeline
(base -> SFT -> DPO/GRPO) and print the answers side by side, so you can *see*
what each stage changes. Reuses the generation helper from recipes/main.py.

=== Usage ===
python -m recipes.compare "What is 2+2?"
python -m recipes.compare "What is 2+2?" --checkpoints gpt2 checkpoints/sft
"""

import argparse

from recipes.main import load_agent

# Default lineup: base model, then each stage's checkpoint as it gets trained.
DEFAULT_CHECKPOINTS = ["gpt2", "checkpoints/sft"]


def compare(question, checkpoints=DEFAULT_CHECKPOINTS, device="cpu"):
    """TODO: load each checkpoint, generate an answer, print them side by side."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Compare answers across training stages")
    parser.add_argument("question", help="Question to ask every checkpoint")
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS,
                        help="Checkpoint paths or HF names, in pipeline order")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    compare(args.question, checkpoints=args.checkpoints, device=args.device)


if __name__ == "__main__":
    main()
