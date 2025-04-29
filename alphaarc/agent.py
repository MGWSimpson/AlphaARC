from alphaarc.logger import make_episode_log
from alphaarc.env import BaseEnv
from alphaarc.policy import BasePolicy

class Agent():
    def __init__(self, policy: BasePolicy):
        pass

    
    def execute_episode(self, env: BaseEnv ):
        episode_log = make_episode_log(env.task.task_key)
        terminated = False
        state = env.reset()

        while not terminated:
            action = self.policy.get_action()
