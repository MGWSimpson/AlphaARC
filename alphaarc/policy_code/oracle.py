import numpy as np 

class Oracle:

    def __init__(self, n_simulations: int, env, encoder_output):
        self.n_simulations = n_simulations
        self.env = env
        import torch
        self.encoder_output = torch.tensor(self.env.tokenized_task).unsqueeze(0)


    def run(self, model, state): 
        actions, child_key_values = model.predict(self.encoder_output, state, past_key_values=None)
        actions_decoded = self.env.tokenizer.batch_decode(actions, skip_special_tokens=True)
        action = None
        decoded_action = None
        for i, act in enumerate(actions_decoded):
            if act in self.env.task.program_lines:
                action = actions[i]
                decoded_action = act

        if len(self.env.task.program_lines.split("\n")) == 1:
            print(actions_decoded)
            print(self.env.task.program_lines)
        return action, actions
        