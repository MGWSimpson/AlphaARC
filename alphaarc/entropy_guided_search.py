
from transformers import T5ForConditionalGeneration, AutoTokenizer
from alphaarc.task import Task
import torch
import matplotlib.pyplot as plt
import numpy as np 
import json
from alphaarc.policy.tokenize import tokenize_task
import heapq, math, time
import torch.nn.functional as F
import math
import os 
from typing import List, Tuple, Optional

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def encode_task(task, tokenizer, model, input_state_max=256, n_examples=10, max_length=256): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task

def entropy_bits(logits: torch.Tensor) -> float:
    if logits.dim() == 2:
        logits = logits.squeeze(0)
    logp = F.log_softmax(logits, -1)
    p    = logp.exp()
    return (-(p * logp).sum() / math.log(2)).item()


def forward_step(model, input_ids, last_ids: torch.Tensor):
    out = model(
        input_ids=input_ids,
        decoder_input_ids=last_ids.unsqueeze(0),
        use_cache=False,

    )
    return out.logits[:, -1, :]

class Node:
    def __init__(self, log_ps, ancestors):
        self.log_ps = log_ps
        self.ancestors = ancestors


class EntropyGuidedSearch:
    def __init__(self,
                 tau,
                 top_k,
                 node_cap, # total forward passes
                 max_length,
                 ):
        
        self.tau = tau
        self.top_k = top_k
        self.node_cap = node_cap
        self.max_length = max_length
    


    def generate_sequences(self, input_ids, model, tok): 

        model.eval()

        finished = []
        visited_nodes = 0

        beam: List[Node] = [(0.0, torch.tensor([0], device='cuda'))]

        while visited_nodes < self.node_cap:

            neg_lp, ids = heapq.heappop(beam)            
            
            # greedy decoding!
            while len(ids) < self.max_length:
                logits  = forward_step(model, input_ids, ids[-1:] )
                H = entropy_bits(logits)

                if H <= self.tau:
                    nxt = logits.argmax(-1)
                    neg_lp -= F.log_softmax(logits, -1)[0, nxt].item()
                    ids = torch.cat([ids, nxt], 0)
                    if nxt.item() == tok.eos_token_id:
                        break
                    continue
                break  
            
            if ids[-1].item() == tok.eos_token_id or len(ids) >= self.max_length:
                finished.append((neg_lp, ids))
            continue

            


if __name__ == "__main__":
    model = T5ForConditionalGeneration.from_pretrained('./finetune/2025-05-19_20-26-16/checkpoint-1647')
    tok = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    search_wrapper = EntropyGuidedSearch(tau=1, top_k=128, node_cap=1000, max_length=512)
    task = Task.from_json('./data/training/0ca9ddb6.json')
    input_ids = torch.tensor(encode_task(task, tok, model)).unsqueeze(0)

    answers = search_wrapper.generate_sequences(input_ids.to('cuda'), model.to('cuda'), tok)
    
    print(tok.batch_decode(answers))