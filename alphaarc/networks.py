import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer
import os
import numpy as np
from numpy import inf
from alphaarc.policy.tokenize import tokenize_task

import torch.nn.functional as F


class PolicyValueNetwork(nn.Module): 
    def __init__(self, model_path, tokenizer_path, temperature=0.95, max_length=1024, num_samples=5, input_state_max=1024):
        super().__init__()
        self.model= T5ForConditionalGeneration.from_pretrained(model_path)        
        self.tokenizer =AutoTokenizer.from_pretrained(tokenizer_path)
        
        # TODO: fix this
        self.value = nn.Linear(512, 1)    
        self.device = 'cuda'        

        # model parameters
        self.temperature = temperature
        self.max_length = max_length
        self.num_samples = num_samples
        self.stop_strings =['\n']
        self.input_state_max = input_state_max


    def _clean_outputs(self, output): 
        sequences = output.sequences  # shape: (batch_size, seq_length)
        logits = output.logits        # tensor of shape (batch_size, seq_length, vocab_size)
        
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        batch_size = sequences.size(0)
        clean_sequences = []
        clean_logits = []
        
        for i in range(batch_size):
            # Extract the i-th sequence and its corresponding logits
            seq = sequences[i]          # shape: (seq_length,)
            logit_seq = logits[i]       # shape: (seq_length, vocab_size)
            # Create mask for tokens that are not BOS or EOS
            mask = (seq != bos_id) & (seq != eos_id)
            # Use the boolean mask to filter the sequence and the corresponding logits
            clean_sequences.append(seq[mask])
            clean_logits.append(logit_seq[mask])
            
        return clean_sequences, clean_logits
    
    def _compute_score_from_logits(self, actions, logits): 
        probabilities = F.softmax(logits, dim=-1) 
        chosen_token_probs = probabilities.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        chosen_token_probs = torch.clamp(chosen_token_probs, min=1e-12)
        log_seq_scores = torch.log(chosen_token_probs).sum(dim=-1)
        normalized_scores = F.softmax(log_seq_scores, dim=0)
        return normalized_scores
    

    def _batch_compute_score(self, actions, logits):
        probabilities = F.softmax(logits, dim=-1)
        
        chosen_token_probs = probabilities.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        
        chosen_token_probs = torch.clamp(chosen_token_probs, min=1e-12)
        
        log_seq_scores = torch.log(chosen_token_probs).sum(dim=-1)
        
        normalized_scores = F.softmax(log_seq_scores, dim=1)
        
        return normalized_scores

    def _compute_actions(self, state):
        outputs = self.model.generate(      state,
                                            temperature=self.temperature,
                                            do_sample=True,
                                            max_new_tokens=20,
                                            num_return_sequences=self.num_samples,
                                            return_dict_in_generate=True,
                                            output_logits=True,
                                            stop_strings=self.stop_strings,
                                            tokenizer= self.tokenizer,
                                            use_cache=False) 

        actions, logits = self._clean_outputs(outputs)
        actions = outputs.sequences[:, :-1] 
        logits = outputs.logits
        logits = torch.stack(logits).to(self.device)
        logits = logits.permute(1, 0, 2)
        action_probs = self._compute_score_from_logits(actions=actions, logits=logits)
        return actions, action_probs
    
    def _compute_values(self, state): 
        last_hidden_state = self.model.encoder(input_ids=state, use_cache=False, output_hidden_states=True).hidden_states[-1]
        values = self.value(last_hidden_state)
        values = values.squeeze()
        values = values[-1] # just take the last value prediction
        return values
    
    def _batch_compute_values(self, states): 
        last_hidden_states = self.model.encoder(input_ids=states, use_cache=False, output_hidden_states=True).hidden_states[-1]
        values = self.value(last_hidden_states).squeeze()
        return values[:, -1]
    # predict expects everything as a tensor.
    def predict(self, state): 
        self.eval()
        with torch.no_grad(): 
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            actions, action_probs =  self._compute_actions(state)
            values = self._compute_values(state)
        return actions.cpu().numpy(), action_probs.cpu().numpy(), values.cpu().numpy()
    

    def forward(self, state, actions): 
        
        B, L = state.shape
        B, A, AL = actions.shape

        logits = []
        for i in range(B):
            output_logits = self.model(input_ids=state[i].repeat(A, 1), 
                                decoder_input_ids=actions[i], 
                                use_cache=False).logits
            logits.append(output_logits)

        logits = torch.stack(logits)
        scores = self._batch_compute_score(actions, logits)
        return scores


    def value_forward(self, state):
        return self._batch_compute_values(states=state)
    
if __name__ == "__main__":
    torch.manual_seed(0)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    network = PolicyValueNetwork()
    network
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')

    state = """x1 = objects(I, T, F, F)
    x2 = rbind(bordering, I)
    x3 = compose(flip, x2)
    x4 = mfilter(x1, x3)
    x5 = fill(I, TWO, x4)"""

    print(tokenizer(state, return_tensors='pt'))
    #actions_probs, values = network.predict(state)
    # actions = [x[0] for x in actions_probs]
    # action_probs2 = network.forward(state, actions)
