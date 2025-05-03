

from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import torch.nn.functional as F


class BaseNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def predict(self, task, state, state_attention_masks,  past_key_values):
        raise NotImplementedError

    def encode(self, task, task_attention_mask): 
        raise NotImplementedError


class PolicyValueNetwork(BaseNetwork): 
    def __init__(self, model_path, tokenizer_path, temperature=0.95,num_samples=5, device='cuda'):
        super().__init__()
        self.model= T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.value = nn.Linear(768, 1) # TODO please fix this.
        self.policy = nn.Linear(768, 1)
        self.device = device

        # model parameters
        self.temperature = temperature
        self.num_samples = num_samples
        self.stop_strings =['\n']
        self.n_calls = 0

    def _compute_actions(self, task, state, state_attention_masks, attention_mask,  past_key_values):
        batch_size = task.shape[0] 

        outputs = self.model.generate(      input_ids=task,
                                            attention_mask=attention_mask,
                                            decoder_input_ids   = state,
                                            decoder_attention_mask = state_attention_masks.bool(), 
                                            temperature=self.temperature,
                                            do_sample=True,
                                            max_new_tokens=20,
                                            num_return_sequences=self.num_samples,
                                            return_dict_in_generate=True,
                                            output_logits=True,
                                            stop_strings=self.stop_strings,
                                            tokenizer= self.tokenizer,
                                            use_cache=True,
                                            output_hidden_states= True,
                                            )         
        
        actions = outputs.sequences.view(batch_size, self.num_samples, -1)
        logits = outputs.logits
        new_actions_shape = len(logits)
        actions = actions[:, : , -new_actions_shape:]

        first_hidden_states = outputs.decoder_hidden_states[0][-1] # index into first gen step + last hidden state
        first_hidden_states = first_hidden_states[: , -1, : ]
        first_hidden_states.view(batch_size, self.num_samples, -1)

        final_hidden_states = torch.stack(outputs.decoder_hidden_states[-1])[-1]
        final_hidden_states = final_hidden_states.view(batch_size, self.num_samples, -1)

 
        

        return actions, self._compute_policy(final_hidden_states),self._compute_values(first_hidden_states), past_key_values

    def _compute_values(self, first_hidden_state): 
        return F.tanh(self.value(first_hidden_state))
        
        
    def _compute_policy(self, last_hidden_state):
        return F.softmax(self.policy(last_hidden_state).squeeze(), dim=-1)


    def predict(self, task, state, state_attention_masks, attention_mask, past_key_values):
        with torch.no_grad(): 
            actions, action_probs, values, past_key_values =  self._compute_actions(task, state, state_attention_masks, attention_mask, past_key_values)
        
        return actions, action_probs ,values, None


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

    def encode(self, task, task_attention_mask):
        return self.model.encoder(input_ids=task, attention_mask=task_attention_mask)
    
# come back to this.
class ActionNetwork(BaseNetwork):
    def __init__(self, model_path, tokenizer_path, temperature=0.95,num_samples=5, device='cuda'):
        super().__init__()

        self.model= T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.value = nn.Linear(768, 1) # TODO please fix this.
        self.policy = nn.Linear(768, 1)
        self.device = device

        # model parameters
        self.temperature = temperature
        self.num_samples = num_samples
        self.stop_strings =['\n']
        self.n_calls = 0
    

