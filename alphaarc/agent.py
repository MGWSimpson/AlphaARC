import numpy as np
from networks import PolicyValueNetwork
from mcts import MCTS
from buffers import ReplayBuffer

from alphaarc.policy.environment import execute_candidate_program
from alphaarc.task import Task
from alphaarc.env import LineLevelArcEnv
 
import torch.optim as optim
import torch


import torch.nn.functional as F



def unpack_actions():
    pass

class Agent(): 
    
    def __init__(self, n_eps=10, n_simulations=5):
        # how many episodes to generate per enviroment
        self.n_eps = n_eps
        self.n_simulations = n_simulations
        self.replay_buffer = ReplayBuffer()
        self.model = PolicyValueNetwork()

        self.n_iters = 10
        self.model.to('cuda')

    def execute_episode(self, env): 
        
        state = env.reset()
        train_examples = []
        terminated = False

        while not terminated:
            self.mcts = MCTS(env, self.model, n_simulations=self.n_simulations)
            root = self.mcts.run(self.model, state)

            action_probs = [0 for _ in range(env.get_action_space())]
            actions = []
            for i, (k, v) in enumerate(root.children.items()):
                action_probs[i] = v.visit_count
                actions.append(k)
            
            action_probs = action_probs / np.sum(action_probs)
            action_probs = zip(actions, action_probs)
            
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
        for eps in range(self.n_eps):
            episode_history = self.execute_episode(env)
            self.replay_buffer.add(episode_history)

        self.train()

    
    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        self.model.train()

        for i in range(self.n_iters):
            states, action_probs, values = self.replay_buffer.sample()
            
            # TODO: start from here making it all batchified.
            # action_probs = list zips
            # targets
            target_vs = torch.FloatTensor(np.array(values).astype(np.float64))
            batch_target_pis = []
            actions = []
            for prob_zip in action_probs:
                # Convert the zipped items to a list of (action, probability)
                prob_list = list(prob_zip)
                # Initialize a distribution tensor with zeros
                target_pi = torch.zeros(5, dtype=torch.float32)
                action_list = []
                # Fill in the probabilities for the actions presented.
                for i, (action, prob) in enumerate(prob_list):
                    target_pi[i] = prob
                    action_list.append(action)

                batch_target_pis.append(target_pi)
                actions.append(action_list)
            # Stack to get a tensor of shape: (batch_size, n_actions)
            target_pis = torch.stack(batch_target_pis)
        
            
            # compute output
            predicted_vs = self.model.value_forward(state=states, actions=actions)
            predicted_pi = self.model.forward(states=states, actions=actions)


            policy_loss = F.cross_entropy(predicted_pi, target_pis)
            value_loss = F.mse_loss( predicted_vs, target_vs)

            loss = policy_loss + value_loss # TODO: add in a balancing term.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



if __name__ == "__main__": 
    task = Task.from_json('data/training/67385a82.json')
    env = LineLevelArcEnv(task)
    agent = Agent()
    
    agent.learn(env)