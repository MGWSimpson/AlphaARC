
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
import os
import torch.nn.functional as F
from alphaarc.dsl.primitives import PRIMITIVE_FUNCTIONS
from alphaarc.program_completer import ProgramCompleter, ProgramSampler
import re
import traceback


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


import itertools
node_counter = itertools.count()

from typing import List, Optional
from transformers import AutoTokenizer


global_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")
from typing import List, Optional
from transformers import AutoTokenizer

from typing import List, Optional
from transformers import AutoTokenizer


def format_as_dummy_program(program_lines):
    return f"""def solve_28bf18c6(I):
    {program_lines}"""


def get_first_new_token_after_prefix(
    texts: List[str],
    common_prefix: str,
    tokenizer_name: str = "Salesforce/codet5p-220m"
) -> List[Optional[str]]:

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    prefix_tokens = tokenizer.encode(common_prefix, add_special_tokens=False)
    prefix_len = len(prefix_tokens)

    tokenized_texts = [tokenizer.encode(text, add_special_tokens=False) for text in texts]

    first_new_tokens = []
    for tokens in tokenized_texts:
        if len(tokens) > prefix_len:
            # Get the first new token after the prefix
            new_token = tokens[prefix_len]
            decoded = tokenizer.decode([new_token])
            first_new_tokens.append(decoded)
            
    return first_new_tokens



def get_new_tokens_after_prefix(
    texts: List[str],
    common_prefix: str,
    tokenizer_name: str = "Salesforce/codet5p-220m"
) -> List[Optional[str]]:
 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    prefix_tokens = tokenizer.encode(common_prefix, add_special_tokens=False)
    prefix_len = len(prefix_tokens)

    tokenized_texts = [tokenizer.encode(text, add_special_tokens=False) for text in texts]

    remaining_tokens = [
        tokenizer.decode(tokens[prefix_len:]) if len(tokens) > prefix_len else None
        for tokens in tokenized_texts
    ]

    return remaining_tokens



def get_decoder_suffixes(reference_ids, target_ids_list):
    results = []
    for target_ids in target_ids_list:
        min_len = min(len(reference_ids), len(target_ids))
        for i in range(min_len):
            if reference_ids[i] != target_ids[i]:
                results.append(target_ids[:i + 1])
                break
        else:
            # No difference found in overlapping portion
            # Include one extra token if target is longer
            if len(target_ids) > len(reference_ids):
                results.append(target_ids[:min_len + 1])
            else:
                results.append(target_ids)
    return results



def extract_function_name(expression):
    match = re.search(r'=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', expression)
    return match.group(1) if match else None





def merge_with_overlap(s1, s2):
    max_overlap = 0
    overlap_start = 0

    # Try every suffix of s1 against every prefix of s2
    for i in range(1, min(len(s1), len(s2)) + 1):
        if s1[-i:] == s2[:i]:
            max_overlap = i

    return s1 + s2[max_overlap:]

@dataclass(order=False)
class Node:

    sort_key: float = field(init=False)
    log_p: float = field(compare=False) # may use this later.
    dec_ids: torch.Tensor = field(compare=False) # ancestors
    n_splits: int  = field(compare=False)# sorts by the number of splits

    def __post_init__(self):
        self.sort_key = (self.n_splits, -self.log_p, )
        self.counter = next(node_counter)  # assign insertion order

    def __repr__(self):
        return f"Node(sort_key={self.sort_key}, n_splits={self.n_splits}, program={global_tokenizer.decode(self.dec_ids)})"
    


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


"""
Code which handles non entropy spikes. Simply extend the sequence.
"""
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
                    n_splits = node.n_splits     # unchanged for greedy extension
                )
        heapq.heappush(frontier,  (new_node.sort_key, new_node.counter, new_node))


def handle_entropy_spike(mask, tok, nodes, log_probs, frontier, completer: ProgramCompleter, task): 
    spike_ids = (~mask).nonzero(as_tuple=False).squeeze(-1).tolist()           

    for idx in spike_ids: 

        program = tok.decode(nodes[idx].dec_ids, skip_special_tokens=True)
        partial_line = program.split("\n")[-1]


        try:
            completions = completer.complete(format_as_dummy_program(program), task.training_examples[0]['input'])
        except Exception as e: # must make this stuff quite robust as finding completions on erroneous code is tricky.
            traceback.print_exc()
            continue
        if len(completions) ==0:
            continue
        
        
        completions = [merge_with_overlap(partial_line, x) for x in completions] 



        # convert the completions to tokenized texts.


        completions = [torch.cat((torch.tensor([0]), tok(x, add_special_tokens=False, return_tensors='pt')['input_ids'].view(-1))) for x in completions]
        
        # merged_tokens = get_decoder_suffixes(nodes[idx].dec_ids, completions)


        
        for new_tokens in completions: # enqueue a node 

            new_node = Node(
                log_p   = nodes[idx].log_p + 0,
                dec_ids = new_tokens.to('cuda'),
                n_splits = (nodes[idx].n_splits + 1)      # unchanged for greedy extension
            )

            
            heapq.heappush(frontier,  (new_node.sort_key, new_node.counter, new_node))

    

def entropy_fanout_search_encdec( 
        model: T5ForConditionalGeneration,
        tok: AutoTokenizer,
        prompt_ids: torch.Tensor,
        env: LineLevelArcEnv,
        completer: ProgramCompleter,
        task, 
        time_limit: int, 
        tau: float,   
        max_len: int = 128,
        batch_size: int = 1,
        k=1,
        ): 
    

    device = prompt_ids.device
    model.eval()
    with torch.no_grad(): # first encode the input once.
        enc_out = model.get_encoder()(prompt_ids.unsqueeze(0))

    n_visted = 0

    num_splits = 0
    num_continuations = 0 

    frontier = [(0, 0, Node(   log_p=0.0,
                        dec_ids=torch.tensor([0],device='cuda') ,
                        n_splits=0))]   # queue on to the frontier the starting node. which is the pad token for T5.
    
    start_time = time.time()
    
    while frontier and time.time() - start_time <= time_limit: # whilst there are still nodes to evaluate and its within budget. proceed search.
        nodes = [heapq.heappop(frontier)[2] for i in range(1)] # pop off batch size number of nodes from frontier.
        batched_decoder_ids = batch_decoder_ids(nodes) # batch and pad the nodes decoder ids
        
        logits = fwd_step_encdec(model, enc_out, batched_decoder_ids ) # perform a forward pass with the encoder-decoder
        entropies = entropy_bits(logits) # compute the entropies.

        n_visted +=1



        for i in range(batched_decoder_ids.shape[0]): # a bit random, but we evaluate here.
            reward, terminated = env.evaluate_program(batched_decoder_ids[i].view(-1), should_token_account=False)
            if reward == 1.0: 
                print("SOLVED!")
                return True

        # compute the stats of the sequence
        log_probs   = torch.log_softmax(logits, dim=-1)          # (B, |V|)
        next_tokens = logits.argmax(dim=-1)                      # (B,)
        mask        = entropies < tau                            # (B,) bool
     

        num_continuations += mask.sum().item()
        num_splits += (~mask).sum().item()

        # basically. depending on whether the entropy is too low or too high, we do different things.
        handle_non_entropy_spike(mask, nodes, next_tokens, log_probs, frontier)
        handle_entropy_spike(mask, tok, nodes, log_probs, frontier, completer, task)

    return False # if failed to solve

if __name__ == "__main__": 
    model = T5ForConditionalGeneration.from_pretrained('./finetune-checkpoint/dev-checkpoint')
    tok = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    task = Task.from_json('./data/training/c8f0f002.json')
    input_ids = torch.tensor(encode_task(task, tok, model))

    env = LineLevelArcEnv('Salesforce/codet5p-220m',  10, 512, 512, 10, 50000)

    env.set_task(task)
    sampler   = ProgramSampler(data_path="./data/")
    completer = ProgramCompleter(sampler)
    
    answers = entropy_fanout_search_encdec( model.to('cuda'), 
                                    tok,
                                    input_ids.to('cuda'),
                                    env,
                                    completer,
                                    task,
                                    tau=-1,
                                    time_limit=10000)


