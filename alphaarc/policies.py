from alphaarc.policy_code.alphaproof import AlphaProof
from policy_code.mcts import MCTS
from policy_code.policy_guided_mcts import PolicyGuidedMCTS
from policy_code.alphazero import AlphaZero
from alphaarc.env import ExceededTokenBudget
import numpy as np
import copy


"""
There is a decision here on whether or not I should store the best program.
Going to ere on the side of not, otherwise I lose the best stuff.
"""

class BasePolicy:
    def __init__(self):
        pass
    
    def policy_init(self):
        raise NotImplementedError
    
    def get_action(self, state):
        raise NotImplementedError
    
class AlphaProofPolicy(BasePolicy):
    def __init__(self, model, env, temperature, n_simulations):
        super().__init__()
        self.model = model
        self.env = env 
        self.temperature = temperature
        self.n_simulations = n_simulations        


    def policy_init(self):
        self.encoder_output = self.model.encode(self.env.tokenized_task, self.env.task_length).squeeze()

        
    def get_action(self, state): 
        self.mcts = AlphaProof(self.env , encoder_output=self.encoder_output,  n_simulations=self.n_simulations)
        root = self.mcts.run(self.model, state)
        actions = root.child_actions
        action_probs = [v.visit_count for v in root.children]
        action_probs = action_probs / np.sum(action_probs)
        action = root.select_action(temperature=self.temperature)
        return action, actions, action_probs



class MCTSPolicy(BasePolicy):
    def __init__(self, model, env, temperature, n_simulations):
        super().__init__()
        self.model= model
        self.env = env
        self.temperature = temperature
        self.n_simulations = n_simulations

        self.root = None

    def policy_init(self):
        self.encoder_output = self.model.encode(self.env.tokenized_task, self.env.task_length).squeeze()
        self.root = None

    # what I could do is basically say, like rather than throwing the exception you just check if that is the last
    # action which the program can construct and return if its terminated.
    def get_action(self, state): 
        
        if self.root is None: # generate the tree, save the root
            self.mcts = MCTS(encoder_output=self.encoder_output, env=self.env, n_simulations=self.n_simulations)
            root = self.mcts.run(self.model, state)
            self.root = copy.deepcopy(root)


        # basically generate this on the first try. Once you have generated it then just step through it. 
        actions = self.root.child_actions
        
        action_probs = [v.visit_count for v in self.root.children]
        action_probs = action_probs / np.sum(action_probs)
        action, action_node = self.root.select_action(temperature=self.temperature)
        self.root = copy.deepcopy(action_node)

        
        terminated = np.sum([v.visit_count for v in self.root.children]) == 0

        return action, actions, action_probs, terminated


class PolicyGuidedMCTSPolicy(BasePolicy):
    def __init__(self, model, env, temperature, n_simulations):
        super().__init__()
        
        self.model= model
        self.env = env
        self.temperature = temperature
        self.n_simulations = n_simulations
    

    def policy_init(self):
        self.encoder_output = self.model.encode(self.env.tokenized_task, self.env.task_length).squeeze()
    

    def get_action(self, state): 
        self.mcts = PolicyGuidedMCTS(encoder_output=self.encoder_output, env=self.env, n_simulations=self.n_simulations)
        root = self.mcts.run(self.model, state)
        actions = root.child_actions
        action_probs = [v.visit_count for v in root.children]
        action_probs = action_probs / np.sum(action_probs)
        action = root.select_action(temperature=self.temperature)
        return action, actions, action_probs



class AlphaZeroPolicy(BasePolicy):
    def __init__(self, model, env, temperature, n_simulations):
        super().__init__()
        self.model = model
        self.env = env 
        self.temperature = temperature
        self.n_simulations = n_simulations        


    def policy_init(self):
        self.encoder_output = self.model.encode(self.env.tokenized_task, self.env.task_length).squeeze()

        
    def get_action(self, state): 
        self.mcts = AlphaZero(self.env , encoder_output=self.encoder_output,  n_simulations=self.n_simulations)
        root = self.mcts.run(self.model, state)
        actions = root.child_actions
        action_probs = [v.visit_count for v in root.children]
        action_probs = action_probs / np.sum(action_probs)
        action = root.select_action(temperature=self.temperature)
        return action, actions, action_probs
