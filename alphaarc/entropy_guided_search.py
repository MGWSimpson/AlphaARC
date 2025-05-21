
from alphaarc.env import LineLevelArcEnv
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import heapq, math, time
from typing import List
from dataclasses import dataclass, field
from alphaarc.task import Task
import numpy as np 
from alphaarc.policy.tokenize import tokenize_task
from torch.nn.utils.rnn import pad_sequence

import torch.nn.functional as F

@dataclass(order=True)
class Node:

    sort_key: float = field(init=False)
    log_p: float # may use this later.
    dec_ids: torch.Tensor # ancestors
    n_splits: int # sorts by the number of splits

    def __post_init__(self):
        self.sort_key = self.n_splits

def fwd_step_encdec(
    model: T5ForConditionalGeneration,
    enc_out,                       # encoder_outputs (computed once)
    dec_ids: torch.Tensor,         # full decoder input so far (1‑D)
):
    out = model(
        encoder_outputs=enc_out,
        decoder_input_ids=dec_ids,
    )  
    return out.logits[:, -1, :]   



def entropy_bits(logits: torch.Tensor) -> float:
    logp = F.log_softmax(logits, -1)
    p = logp.exp()
    return (-(p * logp).sum(dim=-1) / math.log(2))

def encode_task(task, tokenizer, model, input_state_max=256, n_examples=10, max_length=256): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task


def batch_decoder_ids(nodes): 
    decoder_ids_list = [x.dec_ids for x in nodes]
    batched_decoder_ids = pad_sequence(decoder_ids_list, batch_first=True, padding_side='left')
    return batched_decoder_ids



def handle_non_entropy_spike(mask, nodes, next_tokens, log_probs, frontier):
    extend_ids = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    for i in extend_ids:
        node      = nodes[i]
        tok_id    = next_tokens[i].item()
        new_log_p = node.log_p + log_probs[i, tok_id].item()

        new_node = Node(
                log_p   = new_log_p,
                dec_ids = torch.cat([node.dec_ids,
                                     torch.tensor([tok_id], device='cuda')]),
                n_splits = node.n_splits      # unchanged for greedy extension
            )
        heapq.heappush(frontier, new_node)


def handle_entropy_spike(): 
    pass

def entropy_fanout_search_encdec( 
        model: T5ForConditionalGeneration,
        tok: AutoTokenizer,
        prompt_ids: torch.Tensor,
        env: LineLevelArcEnv,
        visit_budget: int = 1000,
        tau: float = 0.5,    
        max_len: int = 128,
        batch_size: int = 16

        ): 
    

    # first encode the sequence, need only do this once.
    device = prompt_ids.device
    model.eval()
    with torch.no_grad():
        enc_out = model.get_encoder()(prompt_ids.unsqueeze(0))

    bos_id = tok.pad_token_id
    n_visted = 0

    frontier = [Node(   log_p=0.0,
                        dec_ids=torch.tensor([bos_id], device=device),
                        n_splits=0), Node(   log_p=0.0,
                        dec_ids=torch.tensor([bos_id], device=device),
                        n_splits=0)]   

    while frontier and n_visted < visit_budget: 
        nodes = [heapq.heappop(frontier) for i in range(min(batch_size, len(frontier)))]
        batched_decoder_ids = batch_decoder_ids(nodes)
        logits = fwd_step_encdec(model, enc_out, batched_decoder_ids )
        entropies = entropy_bits(logits)

        
        log_probs   = torch.log_softmax(logits, dim=-1)          # (B, |V|)
        next_tokens = logits.argmax(dim=-1)                      # (B,)
        mask        = entropies < tau                            # (B,) bool
        

        handle_non_entropy_spike(mask, nodes, next_tokens, log_probs, frontier)
        
        handle_entropy_spike()


if __name__ == "__main__": 
    model = T5ForConditionalGeneration.from_pretrained('./finetune/2025-05-19_20-26-16/checkpoint-1647')
    tok = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    task = Task.from_json('./data/training/0ca9ddb6.json')
    input_ids = torch.tensor(encode_task(task, tok, model))

    env = LineLevelArcEnv('Salesforce/codet5p-220m',  10, 512, 512, 10, 50000)

    print(task.program_lines)
    answers = entropy_fanout_search_encdec( model.to('cuda'), 
                                    tok,
                                    input_ids.to('cuda'),
                                    env)
    
    # print(tok.batch_decode(answers))


