"""
Basically we use entropy to control the branching but then we use a traditional search algorithm to deal with just trusting the log probs.
"""


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
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


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
    tokenizer_name: str = "Salesforce/codet5p-220m") -> List[Optional[str]]:
 
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
    prev_dec_ids: torch.Tensor = field(compare=False)

    def __post_init__(self):
        length_normalized_log_p = self.log_p / self.dec_ids.shape[-1]
        self.sort_key = (- length_normalized_log_p)
        self.length_normalized_log_p = length_normalized_log_p
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


# lazy way to do this for now.
def compute_log_prob(input_, ids): 

    ids = ids.unsqueeze(0)

    with torch.no_grad():
        logits = model(input_, labels=ids).logits              # (B, L, V)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Gather log p for the correct token at each position
        token_logp = log_probs.gather(
            dim=-1, index=ids.unsqueeze(-1)
        ).squeeze(-1)                                # (B, L)

       

        return token_logp.sum(dim=-1).item()




def compute_log_probs_batched(input_batch, ids_batch):
    
    with torch.no_grad():
        labels = ids_batch.clone()

        labels[labels == 0] = -100
        mask = (labels != -100)
        labels[:, 0] = 0

        logits = model(input_ids=input_batch.repeat(ids_batch.shape[0], 1), labels=labels).logits  # (B, L, V)
        log_probs = torch.log_softmax(logits, dim=-1)

        token_logp = log_probs.gather(dim=-1, index=ids_batch.unsqueeze(-1)).squeeze(-1)  # (B, L)
        token_logp = token_logp * mask
        return token_logp.sum(dim=-1)  # (B,)



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


def handle_non_entropy_spike(mask, nodes, next_tokens, log_probs, frontier, tok):
    extend_ids = mask.nonzero(as_tuple=False).squeeze(-1).tolist()


    for i in extend_ids:
        node      = nodes[i]
        
        program = tok.decode(nodes[i].dec_ids, skip_special_tokens=True)
        partial_line = program.split("\n")[-1]


        try:
            completions = completer.complete(format_as_dummy_program(program), task.training_examples[0]['input'])
        except Exception as e: # must make this stuff quite robust as finding completions on erroneous code is tricky.
            traceback.print_exc()
            continue
            
        

        
        

        for k in next_tokens[i]: 
            tok_id    = k.item()
            
            # check to make sure that the completion is within the top; 
            top_k_str = tok.decode(tok_id, skip_special_tokens=True)
            line_check = partial_line + top_k_str
            is_valid = any(valid.startswith(line_check) for valid in completions)
            if not is_valid:
                continue

            
            new_node = Node(
                        log_p   =  node.log_p + log_probs[i, tok_id].item(),
                        dec_ids = torch.cat([node.dec_ids,
                                            torch.tensor([tok_id], device='cuda')]),
                        n_splits = node.n_splits,
                        prev_dec_ids= node.prev_dec_ids    # unchanged for greedy extension
                    )
            heapq.heappush(frontier,  (new_node.sort_key, new_node.counter, new_node))


def handle_entropy_spike(mask, tok, nodes, log_probs, frontier, completer: ProgramCompleter, task, task_encoded): 
    spike_ids = (~mask).nonzero(as_tuple=False).squeeze(-1).tolist()           

    for idx in spike_ids: 
        

        program = tok.decode(nodes[idx].dec_ids, skip_special_tokens=True)
        prev_program = program.split("\n")[:-1]
        partial_line = program.split("\n")[-1]



        try:
            completions = completer.complete(format_as_dummy_program(program), task.training_examples[0]['input'])
        except Exception as e: # must make this stuff quite robust as finding completions on erroneous code is tricky.
            traceback.print_exc()
            continue
        
        if len(completions) ==0:
            continue
        
        


        prev_program_str = "\n".join(prev_program)
        prev_program_str = prev_program_str + "\n"

        
        completions = [merge_with_overlap(partial_line, x) for x in completions]         
        
        if len(prev_program) != 0:
            completions = [ prev_program_str + x for x in completions]


        completions = [torch.cat((torch.tensor([0, 1]), tok(x, add_special_tokens=False, return_tensors='pt')['input_ids'].view(-1))) for x in completions]
        completions_batched = pad_sequence(completions, batch_first=True, padding_value =0, padding_side='right')

        log_ps = compute_log_probs_batched(task_encoded.to('cuda'), completions_batched.to ('cuda'))
        
    
        for i, new_tokens in enumerate(completions):
            new_node = Node( 
                log_p   =   nodes[idx].log_p + log_ps[i],
                dec_ids = new_tokens.to('cuda'),
                n_splits = (nodes[idx].n_splits + 1),
                prev_dec_ids= nodes[idx].dec_ids 
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
                        dec_ids=torch.tensor([0, 1],device='cuda') ,
                        n_splits=0,
                        prev_dec_ids=torch.tensor([0])))]   # queue on to the frontier the starting node. which is the pad token for T5.
    
    start_time = time.time()
    
    while frontier and time.time() - start_time <= time_limit: # whilst there are still nodes to evaluate and its within budget. proceed search.
        
        nodes = [heapq.heappop(frontier)[2] for i in range(1)] # pop off batch size number of nodes from frontier.
        batched_decoder_ids = batch_decoder_ids(nodes) # batch and pad the nodes decoder ids
        
        logits = fwd_step_encdec(model, enc_out, batched_decoder_ids ) # perform a forward pass with the encoder-decoder
        entropies = entropy_bits(logits) # compute the entropies.

        n_visted +=1

        print(nodes[0].length_normalized_log_p)

        for i in range(batched_decoder_ids.shape[0]): # a bit random, but we evaluate here.
            reward, terminated = env.evaluate_program(batched_decoder_ids[i].view(-1), should_token_account=False)
            


            if reward == 1.0:
                print("SOLVED!")
                return True

       

        log_probs   = torch.log_softmax(logits, dim=-1)
        topk_values, topk_indices = torch.topk(log_probs, k=k, dim=-1)  # Get top-k log-probabilities and their indices
        mask        = entropies < tau

        num_continuations += mask.sum().item()
        num_splits += (~mask).sum().item()

        # basically. depending on whether the entropy is too low or too high, we do different things.

        handle_non_entropy_spike(mask, nodes, topk_indices, log_probs, frontier, tok)
        handle_entropy_spike(mask, tok, nodes, log_probs, frontier, completer, task, prompt_ids.unsqueeze(0))
        

    return False # if failed to solve

if __name__ == "__main__": 
    model = T5ForConditionalGeneration.from_pretrained('./finetune/2025-05-27_17-42-37/checkpoint-1650')
    tok = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    task = Task.from_json('./data/training/c8f0f002.json')
    input_ids = torch.tensor(encode_task(task, tok, model))

    env = LineLevelArcEnv('Salesforce/codet5p-220m',  10, 512, 512, 10, 50000)
    
    env.set_task(task)
    sampler   = ProgramSampler(data_path="./data/")
    completer = ProgramCompleter(sampler)
    

    time_limit = (60 * 5)

    answers = entropy_fanout_search_encdec( model.to('cuda'), 
                                    tok,
                                    input_ids.to('cuda'),
                                    env,
                                    completer,
                                    task,
                                    tau=0.5,
                                    time_limit=time_limit)



import math
import copy
import heapq
import random
from typing import List, Optional, Tuple
import numpy as np
import torch

def uct_score(parent: "Node", child: "Node", exploration_c: float = 1.0) -> float:
    if child.visit_count == 0:
        return math.inf
    exploitation = child.value()
    exploration = exploration_c * math.sqrt(math.log(parent.visit_count) / child.visit_count)
    return exploitation + exploration



class Node:
    def __init__(self, parent: Optional["Node"] = None, action_from_parent=None):
        self.parent: Optional[Node] = parent
        self.action_from_parent = action_from_parent

        # statistics
        self.visit_count: int = 0
        self.value_sum: float = 0.0

        # tree structure
        self.child_actions: Optional[List] = None
        self.children: List[Node] = []

        # environment/model caches
        self.state = None
        self.child_key_values = None


    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


    def expand(self, state, actions, child_key_values):
        self.state = copy.deepcopy(state)
        self.child_actions = copy.deepcopy(actions)
        self.child_key_values = None  # or copy.deepcopy(child_key_values)

        self.children = []
        for act in actions:
            self.children.append(Node(parent=self, action_from_parent=act))




class FrontierMCTS:

    def __init__(self, n_simulations: int, exploration_c: float = 1.0):
        self.n_simulations = n_simulations
        self.exploration_c = exploration_c
        self._frontier: List[Tuple[float, float, Node]] = []

  

    def _priority(self, parent: Node, child: Node) -> float:
        return uct_score(parent, child, self.exploration_c)

    def _push_frontier(self, parent: Node, child: Node):
        pri = self._priority(parent, child)
        heapq.heappush(self._frontier, (-pri, random.random(), child))

    def _backpropagate(self, node: Node, value: float):
        curr = node
        while curr is not None:
            curr.visit_count += 1
            curr.value_sum += value
            curr = curr.parent

    def _rollout(self, model, state, actions):
        return random.randint(0, 10) 



    def _simulate_env(self, action, state ): 
        # next_state, value, terminated 
        
        terminated = False
        value = 0 
        next_state = np.concatenate((state, action))

        return next_state, value, terminated

    def run(self, model, prompt, root_state):
        root = Node()
        actions, child_key_values = model.predict(prompt, root_state, past_key_values=None)
        root.expand(root_state, actions, child_key_values)
        # Push every root child into the frontier *once*.
        for child in root.children:
            self._push_frontier(root, child)

        simulation_counter = 0
        while simulation_counter < self.n_simulations:
            _, _, leaf = heapq.heappop(self._frontier)
            parent = leaf.parent

            if leaf.state is None:
                next_state, value, terminated = self._simulate_env(action=leaf.action_from_parent, state=parent.state)
                leaf.state = next_state
            else:
                # We have already generated the state in a previous visit.
                next_state = leaf.state
                terminated = False  # we would have stopped expanding otherwise
                value = None  # will be set later

            if terminated:
                self._backpropagate(leaf, value)
                simulation_counter += 1
                continue

            actions, child_key_values = model.predict(prompt, next_state, past_key_values=None)
            rollout_value = self._rollout(model, next_state, actions)

            leaf.expand(next_state, actions, child_key_values)

            for child in leaf.children:
                self._push_frontier(leaf, child)

            self._backpropagate(leaf, rollout_value)
            simulation_counter += 1

        return root

