import numpy as np
from networks import PolicyValueNetwork
from mcts import MCTS
from buffers import ReplayBuffer



 
class Agent(): 
    
    def __init__(self):
        # how many episodes to generate per enviroment
        self.n_eps = 10
        self.n_simulations = 5
        self.replay_buffer = ReplayBuffer()
        self.model = PolicyValueNetwork()


    def execute_episode(self, env): 
        
        state = env.reset()
        train_examples = []
        terminated = False

        while not terminated:
            self.mcts = MCTS(env, self.model, n_simulations=self.n_simulations)
            root = self.mcts.run(self.model, state)

            action_probs = [0 for _ in range(self.env.get_action_space())]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((state, action_probs))

            action = root.select_action(temperature=0)
            state, reward, terminated = env.step(action=action, state=state)

            if terminated:
                ret = []
                for hist_state,  hist_action_probs in train_examples:
                    # [state, actionProbabilities, Reward]
                    ret.append((hist_state, hist_action_probs, reward))
                return ret


    def learn(self, env): 
        for eps in self.n_eps: 
            episode_history = self.execute_episode(env)
            self.replay_buffer.add(episode_history)

        self.train()

    
