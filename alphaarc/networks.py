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
        
        self.model= T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
        self.model.eval()
        self.value = nn.Linear(512, 1)    
        self.tokenizer =AutoTokenizer.from_pretrained('Salesforce/codet5-small')

        # model parameters
        self.temperature = 0.95
        self.max_length = 1024
        self.num_samples = 5
        self.stop_strings =['\n']

    def _compute_action_probs(self, input_ids, probs ):
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

        batch = []
        for input_sentence, input_probs in zip(input_ids, gen_probs):
            log_probs = 0
            for token, p in zip(input_sentence, input_probs):
                if token not in self.tokenizer.all_special_ids:
                    log_probs += p.item()
            batch.append(log_probs)
        return batch


    def forward(self, state, actions): 
        self.eval()
        new_states = concat_states_and_actions(state, actions)
        logits = self.model.forward(input_ids=new_states, decoder_input_ids=new_states).logits
        
        # print(logits[-1])
        #logits = logits[: , -actions.shape[1]:, :]
        #action_probs = self._compute_action_probs(actions, logits)
        # return action_probs
    

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
            


        

        actions = outputs['sequences'][:, 1: ] # TODO: check if I should keep this.
        logits = outputs.logits
        logits = torch.stack(logits )
        logits = logits.permute(1, 0, 2)

        print(logits)
        self.model.eval()
        new_states = concat_states_and_actions(state, actions)
        outputs = self.model.generate( new_states,
                                        temperature=1.0,
                                            do_sample=True,
                                            max_length=new_states.shape[1]-1,
                                            num_return_sequences=self.num_samples,
                                            return_dict_in_generate=True,
                                            output_logits=True,
                                            stop_strings=self.stop_strings,
                                            tokenizer= self.tokenizer,
                                            use_cache=False) 

        logits = outputs.logits
        logits = torch.stack(logits )
        logits = logits.permute(1, 0, 2)
        print(logits)
        #action_probs = self._compute_action_probs(actions[:, 1:], logits)
        #return actions, action_probs
        return actions
    
    def _compute_values(self, state, actions): 
        new_states = concat_states_and_actions(state, actions)
        last_hidden_state = self.model.forward(input_ids=new_states, decoder_input_ids=new_states,output_hidden_states=True).decoder_hidden_states[-1]
        #values = self.value(last_hidden_state)
        #return values
  

    # inference mode
    def predict(self, state): 
        self.eval()
        #actions, action_probs = 
        actions = self._compute_actions(state)
        #values = self._compute_values(state, actions)
        #return actions, action_probs, values
        return actions
    
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


    input_ids = input_ids = tokenizer(state, return_tensors='pt').to('cuda')['input_ids']
    
    output = network.model.generate( input_ids,
                                        temperature=1.0,
                                            do_sample=True,
                                            max_length=1024,
                                            num_return_sequences=5,
                                            return_dict_in_generate=True,
                                            output_logits=True,
                                            output_scores=True,
                                            stop_strings=['\n'],
                                            tokenizer=tokenizer,
                                            use_cache=False)


    sequences = output.sequences[: , :-1]
    logits = torch.stack(output.logits)
    logits = logits.permute(1, 0, 2)

    actions = tokenizer.batch_decode(sequences)
    new_state = [state + action for action in actions]
    new_input_ids = tokenizer(new_state, return_tensors='pt').to('cuda')['input_ids']
    other_logits = network.model(input_ids=input_ids.repeat(5, 1), decoder_input_ids=sequences).logits
    
    print(logits[:, -1])
    print(other_logits[:, -1])
    
    

    

    #input_ids = tokenizer(state, return_tensors='pt').to('cuda')
    #actions = network.predict(input_ids)
    #result = network.forward(input_ids, actions)

    
    #result = network.forward(input_ids)
    # B x L 
    # B x L x V 
    
    # input_ids = input_ids['input_ids']
    # result = network.model.forward(input_ids=input_ids, decoder_input_ids=input_ids).logits
    # probs = torch.log_softmax(result, dim=-1).detach()

    # print(input_ids.shape)
    # print(result.shape)
    # print(probs.shape)
    # print(network._compute_action_probs(input_ids, probs))