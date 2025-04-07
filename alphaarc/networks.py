import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer
import os
import numpy as np
from numpy import inf
from utils import observation_tuple_to_string


"""
NOTE: presently we handle all tokenization within network. Thinks are assumed to be passed in as strings.
"""
class PolicyValueNetwork(nn.Module): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model= T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
        self.model.eval()
        self.value = nn.Linear(512, 1)    
        self.tokenizer =AutoTokenizer.from_pretrained('Salesforce/codet5-small')

        # model parameters
        self.temperature = 0.95
        self.max_length = 1024
        self.num_samples = 5
        self.stop_strings =['\n']


    def _tokenize(self, string):
        return self.tokenizer(string, return_tensors='pt').to('cuda')


    def _decode(self, tokens):
        return self.tokenizer.batch_decode(tokens)

    # TODO: fill this function out where we assign the pr of taking each action relative to each other one.    
    def _compute_score_from_logits(self, actions, logits): 
        scores = torch.rand((actions.shape[0]))
        scores = torch.softmax(scores, dim=-1)
        return scores

    def forward(self, state, actions): 
        self.eval()
        state = self._tokenize(state)['input_ids']
        actions = self._tokenize(actions)['input_ids']
        logits = network.model(input_ids=state.repeat(actions.shape[0], 1), decoder_input_ids=actions, use_cache=False).logits
        scores = self._compute_score_from_logits(actions, logits)
        return zip(actions, scores)

    def _compute_actions(self, state):
        outputs = self.model.generate(state ,
                                      temperature=1.0,
                                            do_sample=True,
                                            max_length=self.max_length,
                                            num_return_sequences=self.num_samples,
                                            return_dict_in_generate=True,
                                            output_logits=True,
                                            stop_strings=self.stop_strings,
                                            tokenizer= self.tokenizer,
                                            use_cache=False) 

        actions = outputs.sequences[:, :-1]
        logits = outputs.logits
        logits = torch.stack(logits )
        logits = logits.permute(1, 0, 2)
        action_probs = self._compute_score_from_logits(actions=actions, logits=logits)
        return actions, action_probs
    
    def _compute_values(self, state, actions): 
        last_hidden_state = self.model.forward(input_ids=state.repeat(actions.shape[0], 1), decoder_input_ids=actions, use_cache=False, output_hidden_states=True).decoder_hidden_states[-1]
        values = self.value(last_hidden_state)
        return values
  

    def predict(self, state): 
        self.eval()
        state = observation_tuple_to_string(state)
        state = self._tokenize(state)
        state = state['input_ids']
        actions, action_probs =  self._compute_actions(state)
        values = self._compute_values(state, actions)
        actions = self._decode(actions)
        return zip(actions, action_probs), values
        
if __name__ == "__main__":
    torch.manual_seed(0)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    network = PolicyValueNetwork()
    network.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')

    state = """x1 = objects(I, T, F, F)
    x2 = rbind(bordering, I)
    x3 = compose(flip, x2)
    x4 = mfilter(x1, x3)
    x5 = fill(I, TWO, x4)"""


    actions_probs, values = network.predict(state)
    actions = [x[0] for x in actions_probs]
    action_probs2 = network.forward(state, actions)
