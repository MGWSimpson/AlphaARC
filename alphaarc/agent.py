from alphaarc.logger import make_episode_log
from alphaarc.env import BaseEnv
from alphaarc.policies import BasePolicy


from transformers import T5ForConditionalGeneration, AutoTokenizer


class Agent():
    def __init__(self, policy: BasePolicy, replay_q, trajectory_q):
        self.policy = policy
        self.replay_q = replay_q 
        self.trajectory_q = trajectory_q

        self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    
    def _execute_episode(self, env: BaseEnv ):
        terminated = False
        state = env.reset()
        train_examples = []
        self.policy.policy_init()

        while not terminated:
            action, actions, action_probs = self.policy.get_action(state)
            train_examples.append((state, actions, action_probs))
            state, reward, terminated = env.step(action=action, state=state)
            if terminated:
                ret = []
                solved = (reward == 1.0)
                full_task_and_program = (env.tokenized_task, state)
                for hist_state, hist_actions,  hist_action_probs in train_examples:
                    ret.append(( env.tokenized_task, hist_state, hist_actions, hist_action_probs, reward))

                return ret, solved, full_task_and_program


    def learn(self, env: BaseEnv):
        episode_log = make_episode_log(env.task.task_key)
        
        episode_history, solved, full_task_and_program =  self._execute_episode(env)

        if solved:
            self.replay_q.put((full_task_and_program[0], full_task_and_program[1]))
            
        self.trajectory_q.put(episode_history)

        episode_log['solved'] = float(solved)
        return episode_log

    def evaluate(self, env: BaseEnv): 
        self._execute_episode(env)