import numpy as np
from networks import PolicyValueNetwork
from mcts import MCTS
from buffers import ReplayBuffer
from transformers import T5ForConditionalGeneration, AutoTokenizer

from alphaarc.policy.environment import execute_candidate_program
from alphaarc.task import Task
from alphaarc.env import LineLevelArcEnv

import os
import torch.optim as optim
import torch

import torch.nn.functional as F
from tqdm import tqdm

# save.
class Agent(): 
    
    def __init__(self, n_eps=2, n_simulations=20):
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
            self.mcts = MCTS(env , n_simulations=self.n_simulations)
            root = self.mcts.run(self.model, state)
            

            actions = root.child_actions
            action_probs = [v.visit_count for v in root.children]
            action_probs = action_probs / np.sum(action_probs)
            
            train_examples.append((state, actions, action_probs))

            action = root.select_action(temperature=0.5)
            state, reward, terminated = env.step(action=action, state=state)

            if terminated:
                ret = []
                for hist_state, hist_actions,  hist_action_probs in train_examples:
                    # [state, actions,  actionProbabilities, Reward]
                    ret.append((hist_state, hist_actions, hist_action_probs, reward))
                return ret 



    def learn(self, env): 
        for eps in range(1):
            episode_history = self.execute_episode(env)
            self.replay_buffer.add(episode_history)

        # self.train()

    
    def train(self):
        optimizer = optim.Adam(self.model.parameters())
        self.model.train()

        for i in tqdm(range(self.n_iters)):
            states, actions, action_probs, values = self.replay_buffer.sample()
            for idx in range(len(values)): 
                s, pr, v, = states[idx], action_probs[idx], values[idx]
                actions = []
                target_pis = []
                pr = list(pr)
                for lx in pr:
                    l , x = lx
                    actions.append(l)
                    target_pis.append(x) 

                if len(pr) == 0:
                    if v == 1:
                        print("PLEASE NO!")
                    continue
                    

                target_vs = torch.FloatTensor(np.array(v).astype(np.float64)).to('cuda')
                target_pis = torch.FloatTensor(np.array(target_pis).astype(np.float64)).to('cuda')
                
                predicted_pi = self.model.forward(state=s, actions=actions).to('cuda')
                predicted_vs = self.model.value_forward(state=s).to('cuda')


    
                policy_loss = F.cross_entropy(predicted_pi, target_pis)
                value_loss = F.mse_loss( predicted_vs, target_vs)

                loss = policy_loss + value_loss 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tokenizer =AutoTokenizer.from_pretrained('Salesforce/codet5-small')
    task = Task.from_json('data/training/67385a82.json')
    env = LineLevelArcEnv(task, tokenizer=tokenizer)
    agent = Agent()
    
    agent.learn(env)