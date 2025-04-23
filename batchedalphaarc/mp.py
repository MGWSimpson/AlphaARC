import torch
import torch.multiprocessing as mp
import os

from multiprocessing import Process
from batchedalphaarc.agent import Agent
from batchedalphaarc.curriculum import Curriculum
from queue import Empty
from batchedalphaarc.env import LineLevelArcEnv
from dataclasses import dataclass
from transformers import T5ForConditionalGeneration, AutoTokenizer

from torch.nn.utils.rnn import pad_sequence

from batchedalphaarc.env import LineLevelArcEnv
from batchedalphaarc.curriculum import Curriculum
from batchedalphaarc.agent import Agent
from batchedalphaarc.buffers import ReplayBuffer, TrajectoryBuffer
from batchedalphaarc.logger import Logger
from multiprocessing import Process, Value, Lock

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

@dataclass
class RLTrainingConfig:
    rl_batch_size: int =2

@dataclass
class SupervisedTrainingConfig:
    supervised_batch_size: int = 2

@dataclass
class ModelConfig:
    model_path: str = 'finetune/2025-04-18_12-38-42/model'
    tokenizer_path: str = 'Salesforce/codet5p-220m'
    model_temperature: float = 0.95
    device: str = 'cuda'


@dataclass
class batchedalphaarcConfig:
    rl_training_config: RLTrainingConfig = RLTrainingConfig()
    supervised_training_config: SupervisedTrainingConfig = SupervisedTrainingConfig()
    model_config: ModelConfig = ModelConfig()
    n_actions: int = 5
    n_examples: int = 10
    n_episodes_per_task: int = 1
    n_simulations: int = 10
    action_temperature: float = 1
    seed: int = 0
    max_state_len: int = 1024
    max_task_len: int = 512
    max_action_len: int = 20
    trajectory_buffer_capacity = 100_000
    replay_buffer_capacity: int = 100_000
    train_every: int = 50
    n_epochs: int = 1

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer
import os
import numpy as np
from numpy import inf
from batchedalphaarc.policy.tokenize import tokenize_task
import torch.nn.functional as F
import copy
import time
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

class PolicyValueNetwork(nn.Module): 
    def __init__(self, model_path, tokenizer, temperature=0.95,num_samples=5, device='cuda'):
        super().__init__()
        self.model= T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = tokenizer        
        self.value = nn.Linear(768, 1) 
        self.policy = nn.Linear(768, 1)
        self.device = device

        # model parameters
        self.temperature = temperature
        self.num_samples = num_samples
        self.stop_strings =['\n']
        self.n_calls = 0

    
    def _compute_actions(self, task, state, past_key_values):
        
        batch_size = task.shape[0] 
        outputs = self.model.generate(      encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=task), # TODO: this may introduce a bug!
                                            decoder_input_ids   = state,
                                            temperature=self.temperature,
                                            do_sample=True,
                                            max_new_tokens=20,
                                            num_return_sequences=self.num_samples,
                                            return_dict_in_generate=True,
                                            output_logits=True,
                                            stop_strings=self.stop_strings,
                                            tokenizer= self.tokenizer,
                                            use_cache=True,
                                            output_hidden_states= True
                                            )         
        actions = outputs.sequences.view(batch_size, self.num_samples, -1)
        logits = outputs.logits
        new_actions_shape = len(logits)
        #past_key_values = outputs.past_key_values
        actions = actions[:, : , -new_actions_shape:]


        final_hidden_states = torch.stack(outputs.decoder_hidden_states[-1])[-1]
        final_hidden_states = final_hidden_states.view(batch_size, self.num_samples, -1)


        """ first_hidden_states =  torch.stack(outputs.decoder_hidden_states[0])[-1]
        first_hidden_states = first_hidden_states.view(batch_size, self.num_samples, -1)
        first_hidden_states = first_hidden_states[:, -1, :]"""


        return actions, self._compute_policy(final_hidden_states),self._compute_values(final_hidden_states[:, -1, :]), past_key_values

    def _compute_values(self, first_hidden_state): 
        return F.tanh(self.value(first_hidden_state))
        
        
    def _compute_policy(self, last_hidden_state):
        return F.softmax(self.policy(last_hidden_state).squeeze(), dim=-1)


    def predict(self, task, state, past_key_values):
        with torch.no_grad(): 
            actions, action_probs, values, past_key_values =  self._compute_actions(task, state, past_key_values)
        
        return actions, action_probs ,values, None


class ModelRequester():

    def __init__(self, gpu_request_q, encode_request_q):
        self.gpu_request_q = gpu_request_q
        self.encode_request_q = encode_request_q
        self.read_conn , self.send_conn = mp.Pipe(duplex=False)

    def _make_gpu_request(self, task_data): 
        self.gpu_request_q.put_nowait((task_data, self.send_conn))
        result = self.read_conn.recv()
        actions, action_probs, value, child_key_values = result
        actions, action_probs, value, child_key_values = actions.cpu(), action_probs.cpu(), value.cpu(), child_key_values
        return actions.numpy(), action_probs.numpy(), value.numpy(), child_key_values

    def predict(self, task, state, past_key_values):
        return self._make_gpu_request((torch.tensor(task), torch.tensor(state), past_key_values))

    
    def _make_encode_request(self, task): 
        task = torch.tensor(task).unsqueeze(0)
        self.encode_request_q.put_nowait((task, self.send_conn))
        result = self.read_conn.recv()
        return result.cpu()

    def encode(self, task): 
        return self._make_encode_request(task)

class ModelResponder(): 
    def __init__(self, gpu_request_q, encode_request_q, batch_size, model):
        
        self.gpu_request_q = gpu_request_q
        self.encode_request_q = encode_request_q
        self.batch_size = batch_size
        self.model = model


    def _handle_encode_request(self, request): 
        task, connection = request
        task = task.to(self.model.device)
        with torch.no_grad(): 
            result = self.model.model.encoder(task)
        result = result.last_hidden_state
        connection.send(result)

    # TODO: rewrite this to cope with encoding.
    def serve(self): 
        while True: 
            batch = []
            while len(batch) < self.batch_size:
                try:
                    encode_request = self.encode_request_q.get_nowait()
                    self._handle_encode_request(encode_request)
                except Empty: 
                    pass
                
                try: 
                    request = self.gpu_request_q.get_nowait()
                    batch.append(request)
                except Empty: 
                    pass

            data, connections = zip(*batch)
            # packet everything up. and then pass it to the network class
            
            task, state, past_key_values = zip(*data)
            task = pad_sequence(task, batch_first=True)
            state = pad_sequence(state, batch_first=True)

            task, state = task.to(self.model.device), state.to(self.model.device)
            actions, action_probs ,values, past_key_values = self.model.predict(task, state, past_key_values)
            for i, connections in enumerate(connections):
                connections.send(  ( actions[i], 
                                    action_probs[i], 
                                    values[i],
                                    past_key_values))


def tree_worker_fn(task_q: mp.JoinableQueue, gpu_request_q: mp.Queue, encode_request_q: mp.Queue, config: batchedalphaarcConfig, trajectory_buffer_q: mp.Queue, replay_buffer_q: mp.Queue, tasks_solved, lock): 
    model = ModelRequester(gpu_request_q=gpu_request_q, encode_request_q=encode_request_q)

    agent = Agent(trajectory_buffer_q, replay_buffer_q, model,config.n_episodes_per_task, config.n_simulations, config.action_temperature)
    tokenizer = AutoTokenizer.from_pretrained(config.model_config.tokenizer_path)

    while True:
        task = task_q.get()
        env = LineLevelArcEnv(task, tokenizer, config.n_examples, config.max_task_len, config.max_state_len, config.n_actions)
        result = agent.learn(env) 
        with lock: # update the number of tasks solved
            tasks_solved.value += result

        task_q.task_done()
         




def gpu_worker_fn(model_responder: ModelResponder): 
    model_responder.serve()

    

if __name__ == "__main__": 
    n_tree_workers = 4
    mp.set_start_method('spawn', force=True)

    tasks_solved = Value('i', 0)
    lock = Lock()

    config = batchedalphaarcConfig()
    curriculum = Curriculum(dir_paths=['data/evaluation'])
    curriculum_q = mp.JoinableQueue(maxsize=len(curriculum))
    gpu_request_q = mp.Queue()
    encode_request_q = mp.Queue()

    trajectory_buffer_q = mp.Queue()
    replay_buffer_q = mp.Queue()
    

    replay_buffer = ReplayBuffer()
    trajectory_buffer = TrajectoryBuffer()

    model = PolicyValueNetwork(config.model_config.model_path, 
                               AutoTokenizer.from_pretrained(config.model_config.tokenizer_path),
                               config.model_config.model_temperature,
                               config.n_actions,
                               config.model_config.device)
    
    model = model.to(model.device)
    model_responder = ModelResponder(gpu_request_q=gpu_request_q, encode_request_q=encode_request_q, batch_size=n_tree_workers, model=model)
    gpu_worker = Process(target=gpu_worker_fn, args=(model_responder, ))
    gpu_worker.start()
    tree_workers = [Process(target=tree_worker_fn, args=(curriculum_q, gpu_request_q, encode_request_q, config, trajectory_buffer_q, replay_buffer_q, tasks_solved, lock), daemon=True) for _ in range(n_tree_workers)]



    for epoch in range(config.n_epochs):
        train_every = config.train_every
        full_curriculum = curriculum.generate_curriculum()
        
        for i in range(0, len(full_curriculum), train_every):
            print("starting new chunk!")
            curriculum_chunk = full_curriculum[i:i + train_every]
            for task in curriculum_chunk:
                curriculum_q.put(task, block=True)

            for worker in tree_workers:
                worker.start()

            curriculum_q.join()
            print("HERE!")
            
  




    
    

    




    gpu_worker.kill()
    print("workers done")
    print("all done!")