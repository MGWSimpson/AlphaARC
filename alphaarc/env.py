from alphaarc.policy.environment import execute_candidate_program
from alphaarc.task import Task
from alphaarc.policy.tokenize import tokenize_task, TextEncoder
from transformers import AutoTokenizer
import copy
import numpy as np
import torch

def append_action_to_state(state, action): 
    return state + action

def get_last_var_assignment(lines):
    last_var = ""
    lines = reversed(lines)
    for line in lines:
        split_lines = line.split("=")
        if len(split_lines) > 1: 
            return split_lines[0].strip()

    return last_var

# appends a return statement if it does not return anything
def append_return(program):
    lines = program.split("\n")
    if "O" not in lines[-1]:
        last_var = get_last_var_assignment(lines)
        program += f"\nO = identity({last_var})"
    
    return program 



class LineLevelArcEnv:
    def __init__(self, task: Task, tokenizer):
        self.task = task
        self.n_examples = 100
        self.initial_states = [
                training_example["input"]
                for training_example in task.training_examples[: self.n_examples]
        ]
        self.goal_states = [
                training_example["output"]
                for training_example in task.training_examples[: self.n_examples]
        ]
        self.input_state_max = 1024
        self.max_length = 1024

        self.n_actions = 5 # n lines of code allowed.
        self.tokenizer = tokenizer

        self.new_line_arr = self.tokenizer("\n", return_tensors='np')['input_ids'].squeeze()
        tokenized_task = np.array(tokenize_task(self.task, self.tokenizer, self.n_examples, self.input_state_max, self.max_length)['input_ids'])
        self.tokenized_task = np.concatenate((tokenized_task, self.new_line_arr))

    def _state_tokenize(self, state):
        task, program_lines = state
        program_lines = "\n".join(program_lines)
        task_tokens = torch.tensor(tokenize_task(task, self.tokenizer, self.n_examples, self.input_state_max, self.max_length)['input_ids']).unsqueeze(0) 
        
        if len(program_lines) == 0:
            return task_tokens
        
        program_tokens = self.tokenizer(program_lines, return_tensors='pt')['input_ids'] 
        return torch.cat((task_tokens, program_tokens), dim=-1) 

    def _action_tokenize(self, actions):
        return self.tokenizer(actions,padding='longest', return_tensors='pt')['input_ids']


    def _decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def _encode(self, program): 
        return self.tokenizer( program, return_tensors='np')['input_ids'].squeeze()

    # action = new program tokens
    # state =  previous program tokens 
    def step(self, action, state): 
        observation = np.concatenate((state, action, self.new_line_arr))
        terminated = False
        reward = 0
        program = self._decode(observation)
        for i, st in enumerate(self.initial_states):
            candidate_program = append_return(program)
            output = execute_candidate_program(program_string=candidate_program, program_input=st)
            if output == "Invalid Input": 
                #terminated = True # TODO: change this back to false
                reward -= 0

            if output == self.goal_states[i]:
                # terminated = True
                reward +=1

        # print(program)
        terminated = (reward ==  len(self.initial_states)) or len(program.split("\n")) > 15
        reward /= len(self.initial_states)
        observation = self._encode(program)
        return observation, reward, terminated
    

    def get_action_space(self):
        return self.n_actions
 
    def reset(self):
        return np.array([], dtype=np.int64)


if __name__ == "__main__": 
    task = Task.from_json('data/training/67385a82.json')
    env = LineLevelArcEnv(task)
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
    result = tokenize_task(task, tokenizer, 100, 1024, 1024)
    print(result)