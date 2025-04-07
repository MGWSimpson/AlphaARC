import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer
import os
import numpy as np
from numpy import inf


def concat_states_and_actions(state, actions): 
    state = state['input_ids'] # (1, L)
    state = state.repeat((actions.shape[0], 1))
    new_states = torch.cat((state, actions), dim=-1)
    return new_states

def compute_score_from_logits(actions, logits): 
    pass

# note: will handle the tokenization within the network 
# note: presently scores are unprocessed logits, not too sure what the loss functions will want 
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


    def tokenize(self, string):
        return self.tokenizer(string, return_tensors='pt').to('cuda')


    def forward(self, state, actions): 
        self.eval()
        logits = network.model(input_ids=state.repeat(actions.shape[0], 1), decoder_input_ids=actions, use_cache=False).logits
        logits = logits[:, :-1, :]
        return logits

    def _compute_actions(self, state):
        outputs = self.model.generate(state['input_ids'] ,
                                      temperature=1.0,
                                            do_sample=True,
                                            max_length=self.max_length,
                                            num_return_sequences=self.num_samples,
                                            return_dict_in_generate=True,
                                            output_logits=True,
                                            stop_strings=self.stop_strings,
                                            tokenizer= self.tokenizer,
                                            use_cache=False) 

        actions = outputs.sequences[:, :]
        logits = outputs.logits
        logits = torch.stack(logits )
        logits = logits.permute(1, 0, 2)
        return actions, logits
    
    def _compute_values(self, state, actions): 
        state = state['input_ids']
        last_hidden_state = self.model.forward(input_ids=state.repeat(actions.shape[0], 1), decoder_input_ids=actions, use_cache=False, output_hidden_states=True).decoder_hidden_states[-1]
        values = self.value(last_hidden_state)
        return values
  

    """
    state: 
    """
    def predict(self, state): 
        self.eval()
        actions, action_probs =  self._compute_actions(state)
        values = self._compute_values(state, actions)
        return actions, action_probs, values
        
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


    state = network.tokenize(state)
    actions, action_probs, values = network.predict(state)
    action_probs2 = network.forward(state['input_ids'], actions)


    torch.testing.assert_close(action_probs, action_probs2)
     