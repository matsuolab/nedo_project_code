#reference: https://github.com/lucidrains/st-moe-pytorch/blob/main/st_moe_pytorch/st_moe_pytorch.py

import torch
import torch.nn as nn

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

class NoisyGate(nn.Module):
    def __init__(self, hidden_dim, num_experts, noise_mult=1.0, bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.noise_mult = noise_mult
        self.bias = bias
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=self.bias)
    
    def forward(self, x):
        x = self.gate(x)
        noise = gumbel_noise(x)
        out = x + noise * self.noise_mult
        return out

