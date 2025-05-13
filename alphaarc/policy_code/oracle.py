import numpy as np 

class Oracle:

    def __init__(self, n_simulations: int, env, encoder_output):
        self.n_simulations = n_simulations
        self.env = env
        import torch
        self.encoder_output = torch.tensor(self.env.tokenized_task).unsqueeze(0)


    def run(self, model, state): 
        actions, action_probs, child_key_values = model.predict(self.encoder_output, state, past_key_values=None)
        action = np.random.choice(actions)

        print(actions)

        return action, actions
        