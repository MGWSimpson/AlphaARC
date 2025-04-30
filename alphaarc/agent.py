from alphaarc.logger import make_episode_log
from alphaarc.env import BaseEnv
from alphaarc.policies import BasePolicy

class Agent():
    def __init__(self, policy: BasePolicy):
        
        self.policy = policy
        
    
    def _execute_episode(self, env: BaseEnv ):
        terminated = False
        state = env.reset()
        train_examples = []
        self.policy.policy_init()
        while not terminated:
            action, actions, action_probs = self.policy.get_action(state)
            state, reward, terminated = env.step(action=action, state=state)
            train_examples.append((state, actions, action_probs))

            if terminated:
                ret = []
                solved = (reward == 1.0)
                full_task_and_program = (env.tokenized_task, state)
                for hist_state, hist_actions,  hist_action_probs in train_examples:
                    ret.append(( env.tokenized_task, hist_state, hist_actions, hist_action_probs, reward))

                return ret, solved, full_task_and_program


    def learn(self, env: BaseEnv):
        pass

    def evaluate(self, env: BaseEnv): 
        pass