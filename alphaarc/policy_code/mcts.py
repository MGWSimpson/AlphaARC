
import math


class Node: 
    def __init__(self, parent, action, state):
        pass

"""
Algorithm goes like:
    - is it a leaf node?
        - if no -> choose its child according to UCB formula
    - if this is a leaf node.
    ....

    - rollout is:
        -> loop forever:
            - if S_i is a terminal state
                return S_i
            - Else sample a random action. 
            - S_i = simulate(A_i, S_i)

"""
class MCTS:
    def __init__(self, n_simulations):

        self.n_simulations = n_simulations
    
    
    def _selection(self): 
        pass

    def _expansion(self): 
        pass

    def _simulation(self): 
        pass

    def _backpropagation(self): 
        pass

    def run(self): 
        
        # init the tree.
        root = Node(0)
        actions = self.model.predict() # as our model informs our action space.
        root.expand()
        
        for _ in range(self.n_simulations): 
            
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = node.selection_child()
                search_path.append(node)
            
            parent = search_path[-2]
            state = parent.state

            


        

