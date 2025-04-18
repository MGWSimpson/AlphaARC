import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer
import os
import numpy as np
from numpy import inf
from alphaarc.policy.tokenize import tokenize_task
import torch.nn.functional as F
import copy

class PolicyValueNetwork(nn.Module): 
    def __init__(self, model_path, tokenizer, temperature=0.95,num_samples=5, device='cuda'):
        super().__init__()
        self.model= T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = tokenizer        
        self.value = nn.Linear(768, 1) # TODO: please fix this    
        self.policy = nn.Linear(768, 1)
        self.device = device

        # model parameters
        self.temperature = temperature
        self.num_samples = num_samples
        self.stop_strings =['\n']
        



    def _compute_actions(self, task, state):
        if state.shape == (1,0): # if first token, don't pass in decoder input ids
            state = None
            value = torch.tensor([0])
            embedding =  self.model.encoder(task)
        else:
            embedding = copy.copy(self.embedding)
             
        outputs = self.model.generate(      encoder_outputs=embedding,
                                            decoder_input_ids=state,
                                            temperature=self.temperature,
                                            do_sample=True,
                                            max_new_tokens=20,
                                            num_return_sequences=self.num_samples,
                                            return_dict_in_generate=True,
                                            output_logits=True,
                                            stop_strings=self.stop_strings,
                                            tokenizer= self.tokenizer,
                                            use_cache=False,
                                            output_hidden_states= True
                                            ) 

        actions = outputs.sequences
        logits = outputs.logits
        new_actions_shape = len(logits)
        actions = actions[: , -new_actions_shape:]
        final_hidden_states = torch.stack(outputs.decoder_hidden_states[-1])[-1, :, -1, :]
        

        if state is not None:
            first_hidden_states = torch.stack(outputs.decoder_hidden_states[-1])[-1, -1, -new_actions_shape-1, :].squeeze()
            value = self._compute_values(first_hidden_states)
        
        return actions, self._compute_policy(final_hidden_states), value
    
    

    def _compute_values(self, first_hidden_state): 
        return F.tanh(self.value(first_hidden_state))
        
        
    def _compute_policy(self, last_hidden_state):
        return F.softmax(self.policy(last_hidden_state).squeeze(), dim=-1)


    
    def predict(self, task, state): 
        self.eval()
        with torch.no_grad(): 
            task = torch.tensor(task, device=self.device).unsqueeze(0)
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            actions, action_probs, values =  self._compute_actions(task, state)
        
        return actions.cpu().numpy(), action_probs.cpu().numpy(), values.cpu().numpy()

    def set_task(self, task): 
        self.eval()    
        task = torch.tensor(task, device=self.device).unsqueeze(0)
        self.embedding = self.model.encoder(task) 

    def forward(self, task, state, actions):
        B, A, AL = actions.shape

        values = []
        policies = []        
        for i in range(B): 
            task_i = task[i, ]
            state_i = state[i, : ]
            actions_i = actions[i, : ]

            outputs = self.model.forward(   input_ids=task_i.repeat(A, 1), 
                                            decoder_input_ids=torch.concat((state_i.repeat(A, 1), actions_i), dim=-1), 
                                            use_cache=False,
                                            output_hidden_states=True).decoder_hidden_states 
            
            outputs =  torch.stack(outputs)
            first_hidden_state = outputs[-1, 0, -AL-1, :] 
            last_hidden_states =  outputs [-1, :, -1, :]
            
            v = self._compute_values(first_hidden_state)
            p = self._compute_policy(last_hidden_states)

            values.append(v)
            policies.append(p)

        return torch.stack(policies),  torch.stack( values)
     


        

   
 