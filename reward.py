import torch 
import torch.nn as nn 
import torch.nn.functional as F 

"""
Reward Modeling 

Bradley-Terry model: P(y_w > y_l | x) = σ(r(y_w) - r(y_l))
The loss function: -log σ(r(y_w) − r(y_l))
"""

class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_dim):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, input_ids):
        pass 

def reward(model, linear_head, input_ids):

    logits = 

    -torch.log(torch.sigmoid(reward(y_w) - reward(y_l))) 