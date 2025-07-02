
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

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def encode_task(task, tokenizer, model, input_state_max=256, n_examples=10, max_length=256): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task




# ──────────────────────────────────────────
# utilities
# ──────────────────────────────────────────

def entropy_bits(logits: torch.Tensor) -> float:
    """ logits [V]  or [1,V]  → Shannon entropy (bits) """
    if logits.dim() == 2:
        logits = logits.squeeze(0)
    logp = F.log_softmax(logits, -1)
    p = logp.exp()
    return (-(p * logp).sum() / math.log(2)).item()


def fwd_step_encdec(
    model: PreTrainedModel,
    enc_out,                       # encoder_outputs (computed once)
    dec_ids: torch.Tensor,         # full decoder input so far (1‑D)
):
    """Return logits for next token (encoder–decoder, *no* cache)."""
    out = model(
        encoder_outputs=enc_out,
        decoder_input_ids=dec_ids.unsqueeze(0),
    )  # logits shape [1, L_dec, V]
    return out.logits[:, -1, :]               # logits for next token [1,V]


# ──────────────────────────────────────────
# search node
# ──────────────────────────────────────────
@dataclass(order=True)
class Node:
    sort_key: float = field(init=False)
    neg_logp: float
    dec_ids: torch.Tensor  # decoder tokens so far (1‑D)
    depth: int             # number of spikes expanded so far

    def __post_init__(self):
        self.sort_key = self.neg_logp    # heapq ordering key


# ──────────────────────────────────────────
# core search function
# ──────────────────────────────────────────

def entropy_fanout_search_encdec(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    prompt_ids: torch.Tensor,               # encoder tokens (1‑D)
    *,
    tau: float = 0.5,                       # entropy threshold (bits)
    fan_k: int = 128,                       # children per spike
    max_nodes: int = 4000,                  # global model‑call budget
    max_len: int = 128,
    order_by_logp: bool = True,             # False ⇒ random order queue
) -> List[torch.Tensor]:
 

    device = prompt_ids.device
    model.eval()

    # 1) encode prompt once (no grad)
    with torch.no_grad():
        enc_out = model.get_encoder()(prompt_ids.unsqueeze(0))

    bos_id = tok.pad_token_id

    rng = torch.Generator(device=device)

    frontier: List[Node] = [Node(neg_logp=0.0,
                                 dec_ids=torch.tensor([bos_id], device=device),
                                 depth=0)]
    finished: List[Node] = []

    nodes_used = 0

    while (frontier
           and nodes_used < max_nodes):
        
        node = heapq.heappop(frontier)  if order_by_logp else frontier.pop()

        # greedy decoding loop
        while len(node.dec_ids) < max_len and nodes_used < max_nodes:
            logits = fwd_step_encdec(model, enc_out, node.dec_ids)
            nodes_used += 1
            H = entropy_bits(logits)

            if H <= tau: # if the entropy is below the threshold
                nxt = logits.argmax(-1)
                node.neg_logp -= F.log_softmax(logits, -1)[0, nxt].item()
                node.dec_ids = torch.cat([node.dec_ids, nxt], 0)
                if nxt.item() == tok.eos_token_id:
                    finished.append(node)
                continue
            break  # entropy spike detected
            
        
        print(H)
        # finished branch handled above
        if node.dec_ids[-1].item() == tok.eos_token_id or len(node.dec_ids) >= max_len:
            continue
        
        # if the search should no longer continue 
        if nodes_used >= max_nodes:
            break

        
        # 2) entropy spike – fan out tokens that cover ≥ prob_mass
        prob_mass = 0.90                    # ← target coverage
        logits = fwd_step_encdec(model, enc_out, node.dec_ids)
        logp   = F.log_softmax(logits.squeeze(0), -1)
        p_sort, ix_sort = torch.sort(logp.exp(), descending=True)
        cumsum = torch.cumsum(p_sort, 0)
        cutoff = (cumsum < prob_mass).sum().item() + 1   # +1 to cross mass
        sel_p   = p_sort[:cutoff]
        sel_ix  = ix_sort[:cutoff]

        # (optional) cap fan‑out so worst‑case still bounded
        if fan_k is not None and cutoff > fan_k:
            sel_p, sel_ix = sel_p[:fan_k], sel_ix[:fan_k]

        # convert to log‑prob list in requested order
        if order_by_logp:
            # already sorted by prob mass (highest first)
            lp_list = sel_p.log().tolist()
            ix_list = sel_ix.tolist()
        else:
            perm = torch.randperm(sel_ix.size(0), generator=rng, device=device)
            lp_list = sel_p.log()[perm].tolist()
            ix_list = sel_ix[perm].tolist()


        for lp, ix in zip(lp_list, ix_list):
            child_ids = torch.cat([node.dec_ids, torch.tensor([ix], device=device)], 0)
            heapq.heappush(
                frontier,
                Node(neg_logp=node.neg_logp - lp,
                     dec_ids=child_ids,
                     depth=node.depth + 1)
            )

    # collate outputs
    if finished:
        finished.sort(key=lambda n: n.neg_logp)
        return [n.dec_ids for n in finished]

    # fallback partial sequence
    best_partial = min(frontier, key=lambda n: n.neg_logp)
    return [best_partial.dec_ids]





if __name__ == "__main__":
    model = T5ForConditionalGeneration.from_pretrained('./finetune/2025-05-19_20-26-16/checkpoint-1647')
    tok = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    task = Task.from_json('./data/training/0ca9ddb6.json')
    input_ids = torch.tensor(encode_task(task, tok, model))

    print(task.program_lines)
    answers = entropy_fanout_search_encdec( model.to('cuda'), 
                                    tok,
                                    input_ids.to('cuda'))
    
    print(tok.batch_decode(answers))