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

# note: will handle the tokenization within the network
class PolicyValueNetwork(nn.Module): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model= T5ForConditionalGeneration.from_pretrained('Salesforce/codet5p-220m')
        self.value = nn.Linear(768, 1)    
        self.tokenizer =AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')

        # model parameters
        self.temperature = 0.95
        self.max_length = 1024
        self.num_samples = 5
        self.stop_strings =['\n']

    def _to_tokens_and_logprobs(self, input_ids):
        outputs = self.model(input_ids)
        probs = torch.log_softmax(outputs.logits, dim=-1).detach()

        # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

        batch = []
        for input_sentence, input_probs in zip(input_ids, gen_probs):
            text_sequence = []
            for token, p in zip(input_sentence, input_probs):
                if token not in tokenizer.all_special_ids:
                    text_sequence.append((tokenizer.decode(token), p.item()))
            batch.append(text_sequence)
        return batch
    
    def forward(self, state, actions): 
        self.train()
        new_states = concat_states_and_actions(state, actions)
        return new_states

    
    def _compute_action_probs(self, actions, scores): 
        transition_scores = self.model.compute_transition_scores(actions, scores, normalize_logits=True).cpu()
        output_length = np.sum(transition_scores.numpy()  < 0, axis=1)
        length_penalty = self.model.generation_config.length_penalty
        transition_scores[transition_scores == -float('inf')] = 0 # handle padding tokens
        reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)

        return reconstructed_scores

    def _compute_actions(self, state):
        outputs = self.model.generate(input_ids=state['input_ids'] ,
                                        attention_mask=state['attention_mask'],
                                        do_sample=True,
                                        temperature=self.temperature,
                                        max_length=self.max_length,
                                        num_return_sequences=self.num_samples,
                                        return_dict_in_generate=True,
                                        output_scores=True,
                                        stop_strings=self.stop_strings,
                                        tokenizer= self.tokenizer) 
        


        actions = outputs['sequences'][:, 1:] # TODO: check if I should keep this.
        scores = outputs['scores']
        action_probs = self._compute_action_probs(actions, scores)
        return actions, action_probs

    def _compute_values(self, state, actions): 
        new_states = concat_states_and_actions(state, actions)
        last_hidden_state = self.model.forward(input_ids=new_states, decoder_input_ids=new_states,output_hidden_states=True).decoder_hidden_states[-1]
        values = self.value(last_hidden_state)
        return values
  

    # inference mode
    def predict(self, state): 
        self.eval()
        actions, action_probs = self._compute_actions(state)
        values = self._compute_values(state, actions)
        return actions, action_probs, values

    
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    network = PolicyValueNetwork()
    network.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')

    state = """x1 = objects(I, T, F, F)
    x2 = rbind(bordering, I)
    x3 = compose(flip, x2)
    x4 = mfilter(x1, x3)
    x5 = fill(I, TWO, x4)"""

    input_ids = tokenizer(state, return_tensors='pt').to('cuda')

    actions, action_probs, values = network.predict(input_ids)
    
    result = network.forward(input_ids, actions)

    print(action_probs)
    print(result)
    