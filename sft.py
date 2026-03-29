"""
Supervised Fine-Tuning (SFT)
Paper: Ouyang et al., "Training language models to follow instructions 
       with human feedback" (InstructGPT, 2022) https://arxiv.org/abs/2203.02155

What is SFT? 
Given a high-quality response, make the model more likely to produce those exact tokens. 

Loss = -1/T *  Σ log P(token_t | token_<t)


=== Usage === 
python sft.py                           # train on LIMA (default)
python sft.py --dataset alpaca          # train on Alpaca
python sft.py --max_steps 500 --lr 1e-5 # customize training
"""

import torch 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, Dataset 
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse 
import time 

def simple_sft_loss(model, input_ids):
    """
    Standard causal LM SFT loss 
    """
    model_output = model(input_ids)
    logits = model_output.logits

    # shift 
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    return loss 


# Data
class SFTDataset(Dataset):
    """
    Takes instruction/response pairs and tokenizes them into training examples.

    Each example becomes: "### Instruction:\n{input}\n### Response:\n{output}<eos>"

    Why this format? The model needs a consistent pattern so it learns:
    "when you see an instruction, produce a response." The exact template
    doesn't matter much — consistency does.
    """
    def __init__(self, dataset_name="lima", tokenizer=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load dataset from HuggingFace
        if dataset_name == "lima":
            ds = load_dataset("GAIR/lima", split="train")
            # LIMA format: {"conversations": ["user msg", "assistant msg", ...]}
            # We just grab the first two turns (instruction + response)
            self.examples = [
                {"instruction": row["conversations"][0],
                 "response": row["conversations"][1]}
                for row in ds if len(row["conversations"]) >= 2
            ]
        elif dataset_name == "alpaca":
            ds = load_dataset("tatsu-lab/alpaca", split="train")
            self.examples = [
                {"instruction": row["instruction"],
                 "response": row["output"]}
                for row in ds
            ]
        elif dataset_name == "tiny":
            # Built-in toy dataset for quick testing — no download needed
            self.examples = [
                {"instruction": "What is 2+2?", "response": "2+2 equals 4."},
                {"instruction": "Say hello.", "response": "Hello! How can I help you today?"},
                {"instruction": "What color is the sky?", "response": "The sky is blue."},
                {"instruction": "Name a planet.", "response": "Jupiter is the largest planet in our solar system."},
            ] * 25  # repeat to get 100 examples
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"Loaded {len(self.examples)} examples from {dataset_name}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Called by DataLoader to get one training example.

        Returns a dict with 'input_ids' — the tokenized text as a tensor.
        """
        ex = self.examples[idx]

        # Format into a prompt-response string
        text = f"### Instruction:\n{ex['instruction']}\n### Response:\n{ex['response']}"

        # Tokenize: convert text -> list of integers (token IDs)
        # truncation=True  — cut off if longer than max_length
        # padding="max_length" — pad shorter sequences with pad tokens
        #   (we need uniform length so examples can be stacked into a batch)
        # return_tensors="pt" — return PyTorch tensors
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # tokens has shape [1, seq_len] because of return_tensors="pt",
        # squeeze to [seq_len]
        return {"input_ids": tokens["input_ids"].squeeze(0)}


# Training Loop
def train(model, dataset, epochs=1, batch_size=4, lr=2e-5, device="cpu", max_steps=None):
    """
    Standard training loop. Nothing fancy — just gradient descent on the SFT loss.

    The key hyperparameters:
    - lr (learning rate): how big each update step is.
      Too high → unstable, too low → slow. 2e-5 is standard for fine-tuning.
    - batch_size: how many examples to average gradients over.
      Larger = more stable but uses more memory.
    """
    model.to(device)
    model.train()  # put model in training mode (enables dropout, etc.)

    # DataLoader handles batching and shuffling for us
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # AdamW: the standard optimizer for fine-tuning transformers
    # It's Adam (adaptive learning rates per parameter) + weight decay (regularization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        start = time.time()

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)

            # Forward pass: compute loss
            loss = simple_sft_loss(model, input_ids)

            # Backward pass: compute gradients
            # loss.backward() computes ∂loss/∂param for every parameter
            loss.backward()

            # Update: move each parameter in the direction that reduces loss
            # param = param - lr * gradient
            optimizer.step()

            # Zero gradients: PyTorch accumulates gradients by default,
            # so we need to clear them before the next step
            optimizer.zero_grad()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"  step {step}, loss: {loss.item():.4f}")

            if max_steps and step >= max_steps:
                break

        avg_loss = total_loss / len(loader)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{epochs} — avg loss: {avg_loss:.4f} ({elapsed:.1f}s)")

    return model


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nanoTrain SFT")
    parser.add_argument("--dataset", default="lima", choices=["lima", "alpaca", "tiny"])
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # GPT-2 doesn't have a pad token by default — use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {args.dataset}")
    dataset = SFTDataset(args.dataset, tokenizer)

    print("Starting training...")
    model = train(model, dataset, epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, device=args.device, max_steps=args.max_steps)

    print("Done! Saving to checkpoints/sft/")
    model.save_pretrained("checkpoints/sft")
    tokenizer.save_pretrained("checkpoints/sft")