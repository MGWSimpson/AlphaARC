
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


from alphaarc.dsl.primitives import PRIMITIVE_FUNCTIONS






# return a list of starting id's
def compute_starting_ids_of_functions(): 
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    
    starting_ids = []

    for func in PRIMITIVE_FUNCTIONS:
        x = tokenizer.encode(func, return_tensors='np', add_special_tokens=False)[0]
        starting_ids.append( x[0])

        

    return list(set(starting_ids))


STARTING_IDS_OF_FUNCTIONS = compute_starting_ids_of_functions()



def missing_function(line): 
    LHS = line.split("=")[0]
    RHS = line.split("=")[1]
    return RHS.strip() == ""



@dataclass(order=True)
class Node:

    sort_key: float = field(init=False)
    log_p: float = field(compare=False) # may use this later.
    dec_ids: torch.Tensor = field(compare=False) # ancestors
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


# get the starting IDS of all functions
def handle_missing_function(node, frontier): 
    

    for next_tok in STARTING_IDS_OF_FUNCTIONS: 
        new_node = Node(
                log_p   = 0.0,
                dec_ids = torch.cat([node.dec_ids,
                                     torch.tensor([next_tok], device='cuda')]),
                n_splits = node.n_splits + 1      # unchanged for greedy extension
            )
        
        heapq.heappush(frontier, new_node)

    print(len(frontier))


def handle_entropy_spike(mask, tok, nodes, log_probs, frontier): 
    spike_ids = (~mask).nonzero(as_tuple=False).squeeze(-1).tolist()           


    for idx in spike_ids: 
        
        program = tok.decode(nodes[idx].dec_ids, skip_special_tokens=True)
        last_line = program.split("\n")[-1]

        if missing_function(last_line): 
            handle_missing_function(nodes[idx], frontier)
        else:
            pass

def entropy_fanout_search_encdec( 
        model: T5ForConditionalGeneration,
        tok: AutoTokenizer,
        prompt_ids: torch.Tensor,
        env: LineLevelArcEnv,
        visit_budget: int = 1000,
        tau: float = 1,   
        max_len: int = 128,
        batch_size: int = 1

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
                        n_splits=0)]   

    while frontier and n_visted < visit_budget: 
        nodes = [heapq.heappop(frontier) for i in range(min(batch_size, len(frontier)))]
        batched_decoder_ids = batch_decoder_ids(nodes)
        logits = fwd_step_encdec(model, enc_out, batched_decoder_ids )
        entropies = entropy_bits(logits)

        
        log_probs   = torch.log_softmax(logits, dim=-1)          # (B, |V|)
        next_tokens = logits.argmax(dim=-1)                      # (B,)
        mask        = entropies < tau                            # (B,) bool
        
        print(entropies)

        print(tok.decode(nodes[0].dec_ids))
        handle_non_entropy_spike(mask, nodes, next_tokens, log_probs, frontier)
        
        handle_entropy_spike(mask, tok, nodes, log_probs, frontier)


if __name__ == "__main__": 
    model = T5ForConditionalGeneration.from_pretrained('./finetune/2025-05-19_20-26-16/checkpoint-1647')
    tok = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    task = Task.from_json('./data/training/6fa7a44f.json')
    input_ids = torch.tensor(encode_task(task, tok, model))

    env = LineLevelArcEnv('Salesforce/codet5p-220m',  10, 512, 512, 10, 50000)

    print(task.program_lines)
    answers = entropy_fanout_search_encdec( model.to('cuda'), 
                                    tok,
                                    input_ids.to('cuda'),
                                    env)
    
    # print(tok.batch_decode(answers))


