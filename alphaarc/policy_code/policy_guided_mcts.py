
import math
import copy
import random
import numpy as np



def puct_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        value_score = child.value()
    else:
        value_score = 0

    return value_score + prior_score


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

    def expand(self, state, actions, action_probs, child_key_values): 
        self.state = state.copy()
        self.child_actions = copy.deepcopy(actions)
        self.children = [Node(prior=prob) for prob in action_probs]
        self.child_key_values =  None #copy.deepcopy(child_key_values)

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
                best_child_key_value = None # copy.deepcopy(self.child_key_values).batch_select_indices(i)


        return best_action, best_child, best_child_key_value
    

class PolicyGuidedMCTS:
    def __init__(self, env, encoder_output, n_simulations):
        self.n_simulations = n_simulations
        self.env = env
        import torch
        self.encoder_output = torch.tensor(self.env.tokenized_task).unsqueeze(0)
    

    def _rollout(self, model, next_state, actions): 
        action = random.choice(actions)

        while True:
            next_state, value, terminated = self.env.step(action=action, state=next_state)
            
            if terminated:
                return value
            else: 
                actions, _,  child_key_values = model.predict(self.encoder_output, next_state, past_key_values=None) # TODO: may wish to factor in the action probs
                action = random.choice(actions)

    
    def _backpropagate(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum  += value

    def run(self, model, state): 
        
        # initialize the tree.
        root = Node(0)
        actions, action_probs, child_key_values = model.predict(self.encoder_output, state, past_key_values=None)
        
        root.expand(state, actions, action_probs, child_key_values)

        for _ in range(self.n_simulations): 
            node = root
            search_path = [node]
            
            # SELECT
            while node.expanded():
                action, node, child_key_value = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state

            # expansion 
            next_state, value, terminated = self.env.step(action=action, state=state)
            if not terminated: 
                actions,action_probs, child_key_values = model.predict(self.encoder_output, next_state, past_key_values=None)
                value = self._rollout(model, next_state, actions) # rollout
                node.expand(next_state, actions, action_probs,child_key_values)

            # backprop
            self._backpropagate(search_path, value)
        return root

