import copy 

import torch
import torch.nn.functional as F 

"""
Direct Preference Optimization (DPO)

DPO trains a language model directly from preference pairs:

Prompt
Chosen response
Rejected response 

The goal is to make the policy model prefer the chosen response over the
rejected response more strongly than a frozen reference model does. 

For each preference pair, DPO computes:

policy_log_ratio = log_π_policy(chosen | prompt) - log_π_policy(rejected | prompt)
reference_log_ration = log_π_ref(chosen | prompt) - log_π_ref(rejected | prompt)

The DPO loss is: 

L_DPO =  -log σ(
        beta * (policy_log_ratio - reference_log_ratio)
    )

Unlike trad RLHF, DPO does not train a separate reward model and does not require
a RL optimizer such as PPO. The preference signal is applied directly through the lm's
log probabilities. 

Beta controls how strongly the policy is pushed away from the reference model. 
"""
def train_dpo(
    sft_model, 
    steps=200, 
    bs=64, 
    lr=1e-4, 
    beta=0.1,
    device="cpu",
):  
    # both models begin as identical copies of the SFT model 
    policy = copy.deepcopy(sft_model).to(device)
    ref = copy.deepcopy(sft_model).to(device)

    policy.train()

    # The reference model stays fixed throughout DPO training 
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=lr)

    for step in range(steps):
        # Each batch contains the same prompts paired with:
        # c = chosen/preferred responses 
        # r = rejected/dispreferred responses
        c, r = preference_pair(bs)

        c = move_batch_to_device(c, device)
        r = move_batch_to_device(r, device)

        # Policy sequence log probabilities
        #
        # response_logprobs return [batch, response_length]
        # Summing over response tokens give [batch]
        pi_c = response_logprobs(policy, c).sum(dim=1)
        pi_r = response_logprobs(policy, r).sum(dim=1)


