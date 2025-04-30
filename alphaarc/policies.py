from policy_code.alphazero import AlphaZero

import numpy as np

class BasePolicy:
    def __init__(self):
        pass

    
    def policy_init(self):
        raise NotImplementedError
    
    def get_action(self, state):
        raise NotImplementedError
    
class AlphaZero(BasePolicy):
    def __init__(self, model, env, temperature, n_simulations):
        super().__init__()

        self.model = model
        self.env = env 
        self.temperature = temperature
        self.n_simulations = n_simulations        


    def policy_init(self):
        self.encoder_output = self.model.encode(self.env.tokenized_task).squeeze()

    def get_action(self, state): 
        
        self.mcts = AlphaZero(self.env , encoder_output=self.encoder_output,  n_simulations=self.n_simulations)
        root = self.mcts.run(self.model, state)
        actions = root.child_actions


        action_probs = [v.visit_count for v in root.children]
        action_probs = action_probs / np.sum(action_probs)
        action = root.select_action(temperature=self.temperature)
        
        return action, actions, action_probs
