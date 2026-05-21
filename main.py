"""
Inference / demo entry point for nanoTrain.

After SFT (sft.py) you have a checkpoint in checkpoints/sft/ that has learned
"when you see an instruction, produce a response." This file loads that
checkpoint and actually *uses* it: ask a question, get an answer.

The one thing that matters here: we format the prompt with the EXACT same
template the model saw during training. SFT taught the model the pattern

    ### Instruction:
    {question}
    ### Response:
    {answer}

so at inference we feed it everything up to (and including) "### Response:\n"
and let it generate the rest. Use a different template and the model won't
recognize the pattern it learned.

=== Usage ===
python main.py "What is 2+2?"        # one-shot answer
python main.py                       # interactive REPL
python main.py --checkpoint gpt2     # use the base (un-tuned) model to compare
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Must match the template in sft.py's SFTDataset.__getitem__
PROMPT_TEMPLATE = "### Instruction:\n{question}\n### Response:\n"


def load_agent(checkpoint="checkpoints/sft", device="cpu"):
    """
    Load the tokenizer + model once and return a ready-to-call agent function.

    We load once and reuse because loading weights from disk is slow — you
    don't want to do it on every question.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    # GPT-2 has no pad token; SFT used eos as pad, so we match that here.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()  # turn off dropout — we want deterministic, not training, behavior

    def agent(question: str) -> str:
        return generate(model, tokenizer, question, device=device)

    return agent


@torch.no_grad()  # no gradients needed at inference — saves memory and time
def generate(model, tokenizer, question, device="cpu", max_new_tokens=128,
             temperature=0.7, top_p=0.9):
    """
    Turn a question into an answer.

    Steps:
      1. Wrap the question in the training template.
      2. Tokenize -> feed to model.generate (samples one token at a time).
      3. Decode only the NEW tokens (everything after the prompt) and return them.

    Sampling knobs:
    - temperature: <1 makes the model more confident/repetitive, >1 more random.
    - top_p (nucleus): only sample from the smallest set of tokens whose
      probabilities sum to top_p. Keeps output coherent without being rigid.
    """
    prompt = PROMPT_TEMPLATE.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    # output_ids contains the prompt + the generation; slice off the prompt
    # so we return only what the model actually produced.
    new_tokens = output_ids[0, prompt_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # The model may keep going and hallucinate another turn ("### Instruction:"
    # or a repeated "### Response:"). Cut at the first "###" marker so we
    # return just this one answer.
    answer = answer.split("###")[0]
    return answer.strip()


def main():
    parser = argparse.ArgumentParser(description="nanoTrain inference")
    parser.add_argument("question", nargs="?", default=None,
                        help="A question to answer. Omit for interactive mode.")
    parser.add_argument("--checkpoint", default="checkpoints/sft",
                        help="Path or HF name. Use 'gpt2' to see the un-tuned base model.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Loading {args.checkpoint}...")
    agent = load_agent(args.checkpoint, device=args.device)

    if args.question is not None:
        # One-shot mode
        print(agent(args.question))
    else:
        # Interactive REPL — keep asking until Ctrl-C / Ctrl-D / "exit"
        print("nanoTrain agent ready. Ask a question (Ctrl-C or 'exit' to quit).")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit"}:
                break
            if question:
                print(agent(question))


if __name__ == "__main__":
    main()
