import numpy as np 

class Oracle:

    def __init__(self, n_simulations: int, env, encoder_output):
        self.n_simulations = n_simulations
        self.env = env
        import torch
        self.encoder_output = torch.tensor(self.env.tokenized_task).unsqueeze(0)


    def _check_if_present(self, string, correct_program): 


        if string in correct_program:
            return True

        
        rhs = string.split("=")[1]

        final_line = correct_program.split("\n")[-1]


        if rhs in final_line:
            return True


        return False



    def run(self, model, state): 
        actions, child_key_values = model.predict(self.encoder_output, state, past_key_values=None)
        actions_decoded = self.env.tokenizer.batch_decode(actions, skip_special_tokens=True)
        action = None
        decoded_action = None
        for i, act in enumerate(actions_decoded):
            if self._check_if_present(act,self.env.task.program_lines):
                action = actions[i]
                decoded_action = act

        if len(self.env.task.program_lines.split("\n")) == 1:
            print(actions_decoded)
            print(self.env.task.program_lines)
        return action, actions
        