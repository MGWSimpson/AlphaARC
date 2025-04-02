import torch
import torch.nn as nn 

from transformers import T5ForConditionalGeneration

"""
Policy network, will be a pretrained LLM
"""
class PolicyNetwork(nn.Module): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = T5ForConditionalGeneration.from_pretrained('codet5p-220m')

    def forward(self, x): 
        pass 