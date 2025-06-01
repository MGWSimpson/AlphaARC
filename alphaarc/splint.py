import time
import math
import copy
import numpy as np
import torch
from alphaarc.env import LineLevelArcEnv
from transformers import AutoTokenizer, T5ForConditionalGeneration
from alphaarc.program_completer import ProgramCompleter, ProgramSampler

# -- helpers --
def puct_score(parent, child):
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        value_score = child.value()
    else:
        value_score = 0

    return value_score + prior_score


def entropy_bits(logits):
    pass

# -- previously fan out search logic --
# -- quick note, can add multi core stuff in here -- 
class EntropyModelWrapper:
    def __init__(self, model_path, completer, tokenizer_path="Salesforce/codet5p-220m"):
        self.model = T5ForConditionalGeneration(model_path)
        self.completer = completer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)


    def _fwd_step_encdec(self, enc_out, dec_ids): 
        out = self.model(   encoder_outputs=enc_out,
                            decoder_input_ids=dec_ids)  
        
        return out.logits[:, -1, :]   

    
    
    def _handle_entropy_spike(self): 
        pass

    def _handle_non_entropy_spike(self):
        pass

    """
    Contains the fanout logic. For now, will just focus on having one
    """
    def predict(self, enc_out, state, task): # so this is where I would return it.
        

        logits = self._fwd_step_encdec(enc_out, state)
        entropy = entropy_bits(logits).item()

        if entropy > self.tau:
            return self._handle_entropy_spike()
        else: 
            return self._handle_non_entropy_spike()


        
        

    def encode(self, prompt_ids): 
        with torch.no_grad(): # first encode the input once.
            enc_out = self.model.get_encoder()(prompt_ids.unsqueeze(0))
        return enc_out

    def eval(self):
        self.model.eval()

    def rollout(self, enc_out, next_state):
        
        with torch.no_grad(): 
            output = self.model.generate(encoder_outputs=enc_out, decoder_input_ids=next_state)
        
        return output

class Node: 
    def __init__(self, prior= 0):
        self.prior = prior
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
        self.state = state.copy()
        self.child_actions = copy.deepcopy(actions)
        self.children = [Node(prior=prob) for prob in action_probs]

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children])
        actions = self.child_actions
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action_index = np.random.choice(len(actions), p=visit_count_distribution)
            action = actions[action_index]
        return action

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
            next_action,
            model: EntropyModelWrapper,
            env: LineLevelArcEnv): 
    
    program = model.rollout(state, next_action)
    value = env.evaluate_program(program)
    return value



def backpropagate(path, value):
    for node in reversed(path):    
        node.visit_count += 1
        node.value_sum  += value


def run_search(env: LineLevelArcEnv,
               task,
               prompt_ids,
               model: EntropyModelWrapper,
               time_limit=60):


        
        model.eval()


        with torch.no_grad():
            enc_out = model.encode(prompt_ids)


        start_time = time.time()
        root = Node(0)
        init_state = torch.tensor([0, 1], device='cuda')
        actions, action_probs = model.predict(enc_out, init_state, task) # perform the predictions, but with 
        root.expand(state, actions, action_probs)


        while (time.time() - start_time) < time_limit:
            node = root
            search_path = [node]
            
            # SELECT
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state

            # expansion 
            next_state, value, terminated = env.step(action=action, state=state)
            if not terminated: 
                actions,action_probs = model.predict(enc_out, next_state, task)
                value = rollout(model, next_state, actions) # rollout
                node.expand(next_state, actions, action_probs)

            # backprop
            backpropagate(search_path, value)
        return root







if __name__ == "__main__":
    run_search()