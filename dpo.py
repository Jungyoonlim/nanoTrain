import copy 

import torch
import torch.nn.functional as F 

def train_dpo(
    sft_model, 
    steps=200, 
    bs=64, 
    lr=1e-4, 
    beta=0.1,
    device="cpu",
):  
    # both models begin as identical copies of the SFT model 
    policy = copy.deepcopy(sft_model)
    ref = copy.deepcopy(sft_model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    for step in range(steps):
        c, r = preference_pair(bs)
        pi_c = response_logprobs(policy, c).sum(1)
        pi_r = response_logprobs()