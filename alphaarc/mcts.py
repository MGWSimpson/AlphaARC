import math
import numpy as np
from alphaarc.env import LineLevelArcEnv
import copy 
def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        value_score = child.value()
    else:
        value_score = 0

    return value_score + prior_score


def normalize_actions(): 
    # TODO: decide if I will mask moves here.
        # valid_moves = self.game.get_valid_moves(state)
        # action_probs = action_probs * valid_moves  # mask invalid moves
        # action_probs /= np.sum(action_probs)
    pass

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
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
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None
        for i in range(len(self.children)):
            score = ucb_score(self, self.children[i])
            if score > best_score:
                best_score = score
                best_action = self.child_actions[i]
                best_child = self.children[i]
            
        return best_action, best_child

    def expand(self, state, actions, action_probs):
        self.state = state.copy()
        self.child_actions = copy.deepcopy(actions)
        self.children = [Node(prior=prob) for prob in action_probs]


    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), self.prior, self.visit_count, self.value())


class MCTS:
    def __init__(self, env: LineLevelArcEnv, n_simulations):
        self.env = env
        self.n_simulations = n_simulations

    def run(self, model, state):

        root = Node(0)
        actions, action_probs, value = model.predict(self.env.tokenized_task, state)
        root.expand(state, actions, action_probs)
        
        for _ in range(self.n_simulations):
            node = root
            search_path = [node]
            # SELECT
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state
           
            next_state, value, terminated = self.env.step(action=action, state=state)
            # check if not terminated.
            if not terminated:
                # If the game has not ended:
                # EXPAND
                actions, action_probs, value = model.predict(self.env.tokenized_task, next_state)
                # normalize_actions()
                node.expand(next_state, actions, action_probs)
            self.backpropagate(search_path, value)

        return root

    def backpropagate(self, search_path, value):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value  
            node.visit_count += 1