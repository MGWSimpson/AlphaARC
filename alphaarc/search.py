
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
import statistics
from alphaarc.utils import prepare_output_dir, save_stats_to_file
import argparse
import json

import pyvis
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# -- tree viz --

import itertools, json, pathlib, webbrowser, tempfile, textwrap
import networkx as nx

class TreeRecorder:
    _ids = itertools.count()

    def __init__(self, active=False):
        self.g = nx.DiGraph()
        self.meta = {}   
        self.active = active 

    def new_id(self):
        return next(self._ids)

    def add_node(self, node_id, label):
        self.g.add_node(node_id, label=label)

    def add_edge(self, parent_id, child_id, label=""):
        self.g.add_edge(parent_id, child_id, label=label)

    def to_html(self, path="search_tree.html"):
        from pyvis.network import Network         
        nt = Network(height="800px", width="1200px", directed=True)
        for n, data in self.g.nodes(data=True):
            nt.add_node(n, label=data["label"],
                         title=self.meta.get(n, ""), shape="box")
        for u, v, data in self.g.edges(data=True):
            nt.add_edge(u, v, label=data.get("label", ""))
        nt.show(path)
        webbrowser.open(f"file://{pathlib.Path(path).resolve()}")


# -- end --

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

    prior_score = c_puct * child.prior  * math.sqrt(parent.visit_count) / (child.visit_count + 1)

    return prior_score + value_score 

def encode_task(task, tokenizer, model, input_state_max=256, n_examples=10, max_length=256): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task


def format_as_dummy_program(program_lines):
    return f"""def solve_28bf18c6(I):
    {program_lines}"""


def return_empty_nodes(): 
    return [], []

def merge_with_overlap(s1: str, s2: str) -> str:
    for i in range(min(len(s1), len(s2)), 0, -1):
        if s1[-i:] == s2[:i] and i < len(s2):
            return s1 + s2[i:]
    return s1 + s2



def compute_prior(model, input_batch, ids_batch, batch_size: int = 64,
    device='cuda'):
    if device is None:
        device = next(model.parameters()).device

    input_batch = input_batch.to(device)
    seq_logps = []

    with torch.no_grad():
        for start in range(0, ids_batch.size(0), batch_size):
            end = start + batch_size
            ids_sub = ids_batch[start:end].to(device)              # (b, L_tgt)

            labels = ids_sub.clone()
            labels[labels == 0] = -100                             # ignore padding
            labels[:, 0] = 0                                       # keep BOS
            mask = labels.ne(-100)                                 # (b, L_tgt)

            prompt = input_batch.expand(ids_sub.size(0), -1)       # (b, L_prompt)

            logits = model(input_ids=prompt, labels=labels).logits # (b, L_tgt, V)
            log_probs = torch.log_softmax(logits, dim=-1)

            token_ll = log_probs.gather(-1, ids_sub.unsqueeze(-1)).squeeze(-1)
            seq_logps.append((token_ll * mask).sum(-1))            # (b,)

    return torch.cat(seq_logps, dim=0).cpu()                       # (N,)




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
    
    
    def set_env(self, env):
        self.env = env

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
        
        return output.to('cpu')
    

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

        seq_logp = compute_prior(self.model, prompt_ids, completions_batched)
        priors = torch.softmax(seq_logp, dim=0)                    # (B,)

        return completions, priors


class SplintMCTSMethod(BaseMethod):
    def __init__(self, uses_model, model, tokenizer, completer, tau, k ):
        super().__init__(uses_model) 

        self.model = model
        self.completer = completer
        self.tok = tokenizer
        self.tau = tau
        self.k = k


        # stats

        self.n_entropy_spikes = 0
        self.n_non_entropy_spikes = 0


        self.curr_nb_streak   = 0      # length of the *current* run
        self.nb_streaks = []
        self.collapsed_lens = []


        # env
    
    def _dfs_completer_trusted(
        self,
        state,    
        prior,        
        enc_out,
        task, input_ids,
        depth=0, 
        max_depth=5):


        if state.shape[-1] > 512 or depth > max_depth:
            return [], []
        
        # make a quick check.


        reward, terminated = self.env.evaluate_program(state.squeeze(), should_token_account=False)
        if reward == 1.0:
            return [True], [True]


        # compute from the current state, what are the new actions, note that ctions are like the full programs and priors.
        comps, priors = self._handle_non_entropy_spike(
                            state, enc_out, input_ids, task)
        



        # check here! 
        if not comps:                    
            self.collapsed_lens.append(depth)
            return [], []


        # zip them up 
        pairs = zip(priors.tolist(), comps)

        leaves, logps = [], []


        
        for p, comp in pairs:
            new_state = comp

            # check entropy of the new completion.

            nxt_logits = self._fwd_step_encdec(enc_out, new_state.unsqueeze(0))
            if entropy_bits(nxt_logits).item() > self.tau:  # if its a breakpoint, then we stop and return
                self.collapsed_lens.append(depth + 1)
                leaves.append(new_state)
                logps.append(p)
            else:  # still trusted: dive deeper, first check to see if its a terminated program however.

                # if not terminated program, then dive deeper.
                c, lp = self._dfs_completer_trusted(
                            new_state,
                            p, 
                            enc_out, 
                            task, 
                            input_ids,
                            depth + 1)
                leaves.extend(c)
                logps.extend(lp)

        return leaves, logps

    
    

            
    def _fwd_step_encdec(self, enc_out, dec_ids): 
        with torch.no_grad():
            out = self.model(   encoder_outputs=enc_out,
                                decoder_input_ids=dec_ids.to('cuda')).logits.to('cpu')
            
        return out[:, -1, :]

    
    
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


        log_ps = compute_prior(self.model, input_ids, completions_batched)
        priors = torch.softmax(log_ps, dim=0)      
        
        
        return completions, priors


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
        
        
        log_ps = compute_prior(self.model, input_ids  , completions_batched)
        topk_values, topk_indices = torch.topk(log_ps, k=min(self.k, log_ps.shape[-1]), dim=-1)  # Get top-k log-probabilities and their indices
        completions = [completions[i.item()] for i in topk_indices]

        log_ps = topk_values

        return completions, log_ps




    def predict(self, enc_out, state, task, input_ids): # so this is where I would return it.
        logits = self._fwd_step_encdec(enc_out, state.unsqueeze(0))
        entropy = entropy_bits(logits).item()
        topk_values, topk_indices = torch.topk(logits, k=self.k, dim=-1)  # Get top-k log-probabilities and their indices

        if entropy > self.tau:

            self.n_entropy_spikes +=1
            if self.curr_nb_streak:
                self.nb_streaks.append(self.curr_nb_streak)

            self.curr_nb_streak =0
            comps, probs = self._handle_entropy_spike(state, enc_out, input_ids, task)

        else: 
            self.n_non_entropy_spikes += 1
            self.curr_nb_streak  += 1


            comps, log_ps = self._dfs_completer_trusted(
                           state, 0, enc_out, task, input_ids)
            

            for comp in comps:
                if comp is True:
                    return comps, log_ps
            
            probs = torch.tensor(log_ps)
            probs = torch.softmax(probs, dim=0)                    # (B,)



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
            output = self.model.generate(encoder_outputs=enc_out, 
                                         decoder_input_ids=next_state.unsqueeze(0).to('cuda'))
        
        return output.to('cpu')
    
    def collect_stats(self):

        if self.curr_nb_streak:
            self.nb_streaks.append(self.curr_nb_streak)
        
        return {
            "n_entropy_spikes"     : self.n_entropy_spikes,
            "n_non_entropy_spikes" : self.n_non_entropy_spikes,
            "max_non_bp_streak"    : max(self.nb_streaks),
            "avg_non_bp_streak"    : statistics.mean(self.nb_streaks),
            "median_non_bp_streak" : statistics.median(self.nb_streaks)
        }
    

class Node: 
    def __init__(self,parent, recorder, prior= 0):

        if recorder.active:
            self.id = recorder.new_id()  
            recorder.add_node(self.id, label=f"V={prior:.2f}")

        self.prior = prior
        self.parent = parent
        self.visit_count = 0
        self.value_sum = 0
        self.child_actions = None
        self.children = []
        self.state = None
        self.is_expanded = False

    def expanded(self):
        return self.is_expanded

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expand(self, state, actions, action_probs, recorder, tok):
        state = state.to('cpu')
        actions = [x.to('cpu') for x in actions]
        self.state = state.clone()
        self.child_actions = copy.deepcopy(actions)
        self.children = [Node(parent=self, recorder=recorder, prior=prob) for prob in action_probs]

        self.is_expanded = True

        if recorder.active:
            for act, child, prob in zip(actions, self.children, action_probs):
                act_str = tok.decode(act)
                recorder.add_edge(self.id, child.id, label=f"{prob:.2f} | {act_str}")


    def select_child(self): 
        best_score = -np.inf
        best_action = -1
        best_child = None
        
        score_list = []
        for i in range(len(self.children)):
            score = puct_score(self, self.children[i])
            score_list.append(score)
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

    return reward, program

def backpropagate(path, value):
    for node in reversed(path):    
        node.visit_count += 1
        node.value_sum  += value

def run_search(env: LineLevelArcEnv,
               task, 
               prompt_ids,
               model, 
               time_limit,
               recorder):
    


    stats = {"max_depth": 0, 
             "nodes_expanded":0, 
             "solved_program": None,
             "nodes_traversed": 0,
            "avg_branching_factor": 0 }
    

    if model.uses_model:
        model.eval()
        with torch.no_grad():
            enc_out = model.encode(prompt_ids)
    else:
        enc_out = None
    

    start_time = time.time()
    root = Node(None, recorder,  0)

    init_state = torch.tensor([0,1])
    actions, action_probs = model.predict(enc_out, init_state, task, prompt_ids) # perform the predictions, but with 
    

    for act in actions:
        if act is True:
            return True, stats


    root.expand(init_state, actions, action_probs, recorder, env.tokenizer)
    #while (time.time() - start_time) < time_limit:
    while stats["nodes_expanded"] < time_limit:
        node = root
        search_path = [node]


        while node.expanded():
            action, node = node.select_child()
            search_path.append(node)

        if len(search_path) > stats['max_depth']: # note taking
            stats['max_depth'] = len(search_path)
        
        stats['nodes_traversed'] += len(search_path)

        parent = search_path[-2]
        state = parent.state


        next_state = copy.deepcopy(action) # given how the models work, the actions include the appended state 
        value, terminated = env.evaluate_program(next_state, should_token_account=False)

        if value == 1.0:
            stats['extra'] = model.collect_stats()
            stats['solved_program'] = env.tokenizer.decode(next_state)
            return True, stats
        
        if value == -1.0:
             value = 0 
        
        next_state = torch.tensor(next_state)
        if not terminated: 
            actions, action_probs = model.predict(enc_out, next_state, task, prompt_ids)


            for act in actions:
                if act is True:
                    return True, stats
                
            if len(actions ) != 0:
                value, program = rollout(state, actions, enc_out, model, env, task) # rollout
                if value == 1.0:
                    stats['extra'] = model.collect_stats()
                    stats['solved_program'] = env.tokenizer.batch_decode(program)
                    return True, stats

                    
                node.expand(next_state, actions, action_probs, recorder, env.tokenizer) # check in here.

                stats['nodes_expanded'] += 1
                stats['avg_branching_factor'] += len(node.children)

                if value == -1.0:
                    value = 0

        backpropagate(search_path, value)
    stats['extra'] = model.collect_stats()
    
    return False, stats


def run_experiment( method: BaseMethod,
                    tasks: list[Task],
                    time_limit: int,
                    tok: AutoTokenizer,
                    output_path,
                    ):
    
    metrics = init_metrics()

    recorder = TreeRecorder(active=False)    # import this where you build the tree


    # lets sort the list by program length


    tasks = sorted(tasks, key=lambda task: len(task.program_lines))
    
    for task in tasks:
        torch.cuda.empty_cache()
        input_ids = torch.tensor(encode_task(task, tok, None)).to('cuda')
        env = LineLevelArcEnv('Salesforce/codet5p-220m',  10, 512, 512, 10, 50000)
        env.set_task(task)
        method.set_env(env)

        print(f"starting task: {task.task_key}")
        start_time = time.time()
        solved, stats = run_search(env, task, input_ids,  method, time_limit,recorder)
        print(stats)
        print("SOLVED" if solved else "FAILED")
        metrics.append(track_task_metrics(task.task_key, solved, start_time, extra=stats))
        save_metrics_to_file(metrics, output_path)



def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='alphaarc/configs/search/tg_mcts.yaml')
        

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
    

    tau = 0.2
    k = 4
    limit = 300
    
    
    if config['method'] == "MCTS": 
        method = MCTSMethod(uses_model=False, tok=tok, completer=completer)
    elif config['method'] == "TGMCTS":
        method = TGMCTSMethod(uses_model=True, model=model, tok=tok, completer=completer)
    elif config['method'] == "SPLINTMCTS":
        method = SplintMCTSMethod(uses_model=True, model=model, tokenizer=tok, completer=completer, tau=tau, k=k)
    else:
        raise ValueError("Method does not exist!")


     
    output_dir =  f"results/{config['method'].lower()}-{k}-{tau}-{limit}"
    prepare_output_dir(output_dir)
    pl.seed_everything(0)


    start_time = time.time()
    run_experiment(method=method,
                   tasks=curriculum.generate_curriculum(),
                   time_limit=(limit),
                   tok=tok,
                   output_path=output_dir,
                   )

    print(f"end time: {time.time() - start_time}")

if __name__ == "__main__": 
    main()