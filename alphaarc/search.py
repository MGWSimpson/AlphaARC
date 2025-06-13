
import copy
import time
import math
import copy
import numpy as np
import torch
import os 
from alphaarc.env import LineLevelArcEnv
from transformers import AutoTokenizer, T5ForConditionalGeneration
from alphaarc.program_completer import ProgramCompleter, ProgramSampler
from alphaarc.policy.tokenize import tokenize_task
from torch.nn.utils.rnn import pad_sequence
from alphaarc.task import Task
import torch.nn.functional as F
import traceback
import random
import torch.nn as nn
from alphaarc.configs import load_config, build_curriculum, build_env
from alphaarc.utils import load_key_split, relabel_task
import pytorch_lightning as pl

from alphaarc.utils import prepare_output_dir, save_stats_to_file
import argparse
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -- experiment helpers -- 


def init_metrics():
    return []


def track_task_metrics(task_key, success, start_time, extra=None):
    return {
        "task": task_key,
        "success": success,
        "time_sec": round(time.time() - start_time, 2),
        "extra": extra or {}
    }

def save_metrics_to_file(metrics, output_path):
    os.makedirs(output_path, exist_ok=True)
    path = os.path.join(output_path, "experiment_results.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


# -- end experiment helpers

# -- helpers --

def puct_score(parent, child, c_puct=1):
    if child.visit_count > 0:
        value_score = child.value()
    else:
        value_score = 0

    prior_score = c_puct * child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)

    return value_score + prior_score

def encode_task(task, tokenizer, model, input_state_max=256, n_examples=10, max_length=256): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task


def format_as_dummy_program(program_lines):
    return f"""def solve_28bf18c6(I):
    {program_lines}"""


def return_empty_nodes(): 
    return [], []

def merge_with_overlap(s1, s2):
    max_overlap = 0
    overlap_start = 0

    for i in range(1, min(len(s1), len(s2)) + 1):
        if s1[-i:] == s2[:i]:
            max_overlap = i

    return s1 + s2[max_overlap:]

def compute_log_probs_batched(model, input_batch, ids_batch):
    
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

def entropy_bits(logits):
    logp = F.log_softmax(logits, -1)
    p = logp.exp()
    return (-(p * logp).sum(dim=-1) / math.log(2))


# will need to redistribute the probs
def prune_node(node):
     if node.parent:
        idx = node.parent.children.index(node)
        del node.parent.children[idx]
        del node.parent.child_actions[idx]
        if len(node.parent.children) == 0 and node.parent.expanded:
            prune_node(node.parent)


# -- end helpers --

class BaseMethod:
    def __init__(self, uses_model):
        self.uses_model = uses_model


    def rollout(self, enc_out, action, task): 
        raise NotImplementedError
    
    def predict(enc_out, init_state, task, prompt_ids): 
        raise NotImplementedError 
    
 
    def collect_stats(self): 
        return {}
    
# generates all actions + no priors 
class MCTSMethod(BaseMethod): 
    
    def __init__(self, uses_model, tok,  completer):
        super().__init__(uses_model) 

        self.tok = tok
        self.completer = completer
    

    def rollout(self, enc_out, state, task):
        program = self.tok.decode(state.squeeze(), skip_special_tokens=True)

        completions = [None]
        xtime = time.time()
        while len(completions) > 0: # if we can still generate completions, keep going
            try:
                completions = self.completer.complete(format_as_dummy_program(program), task.training_examples[0]['input'])
                random_action = random.choice(completions) # needs to merge with overlap
                program = program + random_action
            except Exception as e: # must make this stuff quite robust as finding completions on erroneous code is tricky.
                break
        
        program = self.tok.encode(program, add_special_tokens=False, return_tensors='pt')
        return program
        
        
    def predict(self, _, state, task, __):
        program = self.tok.decode(state, skip_special_tokens=True)

        prev_program = program.split("\n")[:-1]
        partial_line = program.split("\n")[-1]



        try:
            completions = self.completer.complete(format_as_dummy_program(program), task.training_examples[0]['input'])
        except Exception as e: # must make this stuff quite robust as finding completions on erroneous code is tricky.
            return return_empty_nodes()
    
        if len(completions) ==0:
            return return_empty_nodes()

        prev_program_str = "\n".join(prev_program)
        prev_program_str = prev_program_str + "\n"

        
        completions = [merge_with_overlap(partial_line, x) for x in completions]         
        
        if len(prev_program) != 0:
            completions = [ prev_program_str + x for x in completions]

        completions = [self.tok(x, add_special_tokens=False, return_tensors='pt')['input_ids'].view(-1) for x in completions]
        priors = [1 for i in range(len(completions))]
        return completions, priors



class TGMCTSMethod(BaseMethod): 
    
    def __init__(self, uses_model, model, tok, completer):
        super().__init__(uses_model)

        self.model = model
        self.tok = tok 
        self.completer = completer

    
    def rollout(self, enc_out, next_state, task):

        with torch.no_grad(): 
            output = self.model.generate(encoder_outputs=enc_out, decoder_input_ids=next_state.unsqueeze(0).to('cuda'))
        
        return output
    

    def eval(self): 
        self.model.eval()


    
    def encode(self, prompt_ids): 
        with torch.no_grad(): # first encode the input once.
            enc_out = self.model.get_encoder()(prompt_ids.unsqueeze(0))
        return enc_out

    def predict(self, enc_out, state, task, prompt_ids):
        program = self.tok.decode(state, skip_special_tokens=True)

        prev_program = program.split("\n")[:-1]
        partial_line = program.split("\n")[-1]



        try:
            completions = self.completer.complete(format_as_dummy_program(program), task.training_examples[0]['input'])
        except Exception as e: # must make this stuff quite robust as finding completions on erroneous code is tricky.
            return return_empty_nodes()
    
        if len(completions) ==0:
            return return_empty_nodes()

        prev_program_str = "\n".join(prev_program)
        prev_program_str = prev_program_str + "\n"

        
        completions = [merge_with_overlap(partial_line, x) for x in completions]         
        
        if len(prev_program) != 0:
            completions = [ prev_program_str + x for x in completions]


        completions = [torch.cat((torch.tensor([0, 1]), 
                                  self.tok(x, add_special_tokens=False, return_tensors='pt')['input_ids'].view(-1))) for x in completions]

        completions_batched = pad_sequence(completions, batch_first=True, padding_value =0, padding_side='right')

        # then compute priors over all.
        priors = compute_log_probs_batched(self.model, prompt_ids .to('cuda'), completions_batched.to ('cuda'))
        priors = F.softmax(priors, dim=-1)
        return completions, priors


class SplintMCTSMethod(BaseMethod):
    def __init__(self, uses_model, model, tokenizer, completer, tau, k):
        super().__init__(uses_model) 

        self.model = model
        self.completer = completer
        self.tok = tokenizer
        self.tau = tau
        self.k = k


        # stats

        self.n_entropy_spikes = 0
        self.n_non_entropy_spikes = 0


    def _fwd_step_encdec(self, enc_out, dec_ids): 
        out = self.model(   encoder_outputs=enc_out,
                            decoder_input_ids=dec_ids)  
        
        return out.logits[:, -1, :]   

    
    
    def _handle_entropy_spike(self, state, enc_out, input_ids, task): 
        
        program = self.tok.decode(state, skip_special_tokens=True)

        prev_program = program.split("\n")[:-1]
        partial_line = program.split("\n")[-1]



        try:
            completions = self.completer.complete(format_as_dummy_program(program), task.training_examples[0]['input'])
        except Exception as e: # must make this stuff quite robust as finding completions on erroneous code is tricky.
            return return_empty_nodes()
    
        if len(completions) ==0:
            return return_empty_nodes()

        prev_program_str = "\n".join(prev_program)
        prev_program_str = prev_program_str + "\n"

        
        completions = [merge_with_overlap(partial_line, x) for x in completions]         
        
        if len(prev_program) != 0:
            completions = [ prev_program_str + x for x in completions]


        completions = [torch.cat((torch.tensor([0, 1]), 
                                  self.tok(x, add_special_tokens=False, return_tensors='pt')['input_ids'].view(-1))) for x in completions]
        completions_batched = pad_sequence(completions, batch_first=True, padding_value =0, padding_side='right')

        log_ps = compute_log_probs_batched(self.model, input_ids .to('cuda'), completions_batched.to ('cuda'))
        log_ps = F.softmax(log_ps, dim=-1)

        return completions, log_ps


    """
    Here, you handle the problem I was running into by accessing the completions logit.
    This just gets around the whole problem of it being or not being a valid completion.
    And retains the whole token efficency thing.    
    """
    def _handle_non_entropy_spike(self, state, enc_out, input_ids, task):    

        program = self.tok.decode(state, skip_special_tokens=True)

        prev_program = program.split("\n")[:-1]
        partial_line = program.split("\n")[-1]



        try:
            completions = self.completer.complete(format_as_dummy_program(program), task.training_examples[0]['input'])
        except Exception as e: # must make this stuff quite robust as finding completions on erroneous code is tricky.
            traceback.print_exc()
            return return_empty_nodes()
    
        if len(completions) ==0:
            return return_empty_nodes()

        prev_program_str = "\n".join(prev_program)
        prev_program_str = prev_program_str + "\n"

        
        completions = [merge_with_overlap(partial_line, x) for x in completions]         
        
        if len(prev_program) != 0:
            completions = [ prev_program_str + x for x in completions]


        completions = [torch.cat((torch.tensor([0, 1]), 
                                  self.tok(x, add_special_tokens=False, return_tensors='pt')['input_ids'].view(-1))) for x in completions]
        completions_batched = pad_sequence(completions, batch_first=True, padding_value =0, padding_side='right')
        log_ps = compute_log_probs_batched(self.model, input_ids .to('cuda'), completions_batched.to ('cuda'))
        topk_values, topk_indices = torch.topk(-log_ps, k=min(self.k, log_ps.shape[-1]), dim=-1)  # Get top-k log-probabilities and their indices
        
        completions = [completions[i.item()] for i in topk_indices]
        probs = F.softmax(topk_values, dim=-1)

        return completions, probs




    def predict(self, enc_out, state, task, input_ids): # so this is where I would return it.
        logits = self._fwd_step_encdec(enc_out, state.unsqueeze(0))
        entropy = entropy_bits(logits).item()
        topk_values, topk_indices = torch.topk(logits, k=self.k, dim=-1)  # Get top-k log-probabilities and their indices

        if entropy > self.tau:
            self.n_entropy_spikes +=1
            comps, probs = self._handle_entropy_spike(state, enc_out, input_ids, task)
        else: 
            self.n_non_entropy_spikes +=1
            comps, probs = self._handle_non_entropy_spike(state, enc_out, input_ids, task)

        # format the returns.
        return comps, probs
        
        

    def encode(self, prompt_ids): 
        with torch.no_grad(): # first encode the input once.
            enc_out = self.model.get_encoder()(prompt_ids.unsqueeze(0))
        return enc_out

    def eval(self):
        self.model.eval()

    # perform a rollout from the current state.
    def rollout(self, enc_out, next_state, task):
        with torch.no_grad(): 
            output = self.model.generate(encoder_outputs=enc_out, decoder_input_ids=next_state.unsqueeze(0).to('cuda'))
        
        return output
    
    def collect_stats(self):
        return {"n_entropy_spikes": self.n_entropy_spikes, "n_non_entropy_spikes": self.n_non_entropy_spikes}
    

class Node: 
    def __init__(self,parent, prior= 0):
        self.prior = prior
        self.parent = parent
        self.visit_count = 0
        self.value_sum = 0
        self.child_actions = None
        self.children = []
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expand(self, state, actions, action_probs):
        state = state.to('cpu')
        actions = [x.to('cpu') for x in actions]
        self.state = state.clone()
        self.child_actions = copy.deepcopy(actions)
        self.children = [Node(parent=self, prior=prob) for prob in action_probs]


    def select_child(self): 
        best_score = -np.inf
        best_action = -1
        best_child = None

        for i in range(len(self.children)):
            score = puct_score(self, self.children[i])
            if score > best_score:
                best_score = score
                best_action = self.child_actions[i]
                best_child = self.children[i]


        return best_action, best_child    

def rollout( state,
            actions,
            enc_out,
            model: BaseMethod,
            env: LineLevelArcEnv,
            task): 

    start_time = time.time ()
    # need to make a random choice of actions    
    action = random.choice(actions)
    program = model.rollout(enc_out, action, task)
    reward, terminated = env.evaluate_program(program.squeeze(), should_token_account=False)
    return reward

def backpropagate(path, value):
    for node in reversed(path):    
        node.visit_count += 1
        node.value_sum  += value

def run_search(env: LineLevelArcEnv,
               task, 
               prompt_ids,
               model, 
               time_limit=60):
    


    stats = {"max_depth": 0}
    if model.uses_model:
        model.eval()
        with torch.no_grad():
            enc_out = model.encode(prompt_ids)
    else:
        enc_out = None
    

    start_time = time.time()
    root = Node(None, 0)

    init_state = torch.tensor([0,1], device='cuda')
    actions, action_probs = model.predict(enc_out, init_state, task, prompt_ids) # perform the predictions, but with 
    root.expand(init_state, actions, action_probs)

    while (time.time() - start_time) < time_limit:
        node = root
        search_path = [node]


        while node.expanded():
            action, node = node.select_child()
            search_path.append(node)

        if len(search_path) > stats['max_depth']: # note taking
            stats['max_depth'] = len(search_path)

        parent = search_path[-2]
        state = parent.state

        next_state = copy.deepcopy(action) # given how the models work, the actions include the appended state 
        value, terminated = env.evaluate_program(next_state, should_token_account=False)

        if value == 1.0:
            return True
        
        next_state = torch.tensor(next_state)
        if not terminated: 
            actions, action_probs = model.predict(enc_out, next_state.to('cuda'), task, prompt_ids)
            # if no further actions
            if len(actions) == 0: 
                value = -1.0
            else:
                value = rollout(state, actions, enc_out, model, env, task) # rollout
                if value == 1.0:
                    return True
                
                node.expand(next_state, actions, action_probs) # check in here.

        backpropagate(search_path, value)
    print(model.collect_stats())
    return False


def run_experiment( method: BaseMethod,
                    tasks: list[Task],
                    time_limit: int,
                    tok: AutoTokenizer,
                    output_path
                    ):
    
    metrics = init_metrics()

    for task in tasks[:1]:
        task = Task.from_json('./data/training/c8f0f002.json')
        input_ids = torch.tensor(encode_task(task, tok, None)).to('cuda')
        env = LineLevelArcEnv('Salesforce/codet5p-220m',  10, 512, 512, 10, 50000)
        env.set_task(task)


        print(f"starting task: {task.task_key}")
        start_time = time.time()
        solved = run_search(env, task, input_ids,  method, time_limit)
        print("SOLVED" if solved else "FAILED")
        metrics.append(track_task_metrics(task.task_key, solved, start_time))

    save_metrics_to_file(metrics, output_path)



def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='alphaarc/configs/search/splint_mcts.yaml')
        

    args = parser.parse_args()
    config = load_config(args.config_path)
    curriculum = build_curriculum(config['training_curriculum_config'])
    config = load_config(args.config_path)
    
    task_key_split = load_key_split('data/split_keys.json')
    curriculum.prune_tasks_not_in_list(tasks_to_keep=task_key_split['val'])
    

    model = T5ForConditionalGeneration.from_pretrained(config['model_path'])
    model.to('cuda')
    tok = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    sampler   = ProgramSampler(data_path="./data/")
    completer = ProgramCompleter(sampler)
    
    
    
    if config['method'] == "MCTS": 
        method = MCTSMethod(uses_model=False, tok=tok, completer=completer)
    elif config['method'] == "TGMCTS":
        method = TGMCTSMethod(uses_model=True, model=model, tok=tok, completer=completer)
    elif config['method'] == "SPLINTMCTS":
        method = SplintMCTSMethod(uses_model=True, model=model, tokenizer=tok, completer=completer, tau=0.3, k=1)
    else:
        raise ValueError("Method does not exist!")


     
    output_dir =  f"results/{config['method'].lower()}"
    prepare_output_dir(output_dir)
    pl.seed_everything(0)
    


    run_experiment(method=method,
                   tasks=curriculum.generate_curriculum(),
                   time_limit=(60 * 3),
                   tok=tok,
                   output_path=output_dir)


if __name__ == "__main__": 
    main()