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
    
    
    """
    def __init__(self, dataset_name="lima", tokenizer=None, max_length=512):
        pass 


    def __len__(self):
        return len(self.examples)
    
    def 


# Training Loop 




# CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nanoTrain SFT")
    