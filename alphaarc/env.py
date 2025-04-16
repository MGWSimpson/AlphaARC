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
    if "O = " not in program:
        last_var = get_last_var_assignment(lines)
        program += f"O = identity({last_var})"
    return program 


NEW_LINE_TOKEN_ID = 203

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

        self.new_line_arr = np.array([NEW_LINE_TOKEN_ID])
        tokenized_task = np.array(tokenize_task(self.task, self.tokenizer, self.n_examples, self.input_state_max, self.max_length)['input_ids'])
        self.tokenized_task = np.concatenate((tokenized_task, self.new_line_arr))
        
    
    def _add_new_line_if_absent(self, action): 
        if NEW_LINE_TOKEN_ID not in action:
            action = np.concatenate((action, self.new_line_arr))
        return action

    def _decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def _encode(self, program): 
        return self.tokenizer( program, add_special_tokens=False, return_tensors='np')['input_ids'].squeeze()

    # action = new program tokens
    # state =  previous program tokens 
    def step(self, action, state): 
        action = self._add_new_line_if_absent(action)
        observation = np.concatenate((state, action))
        terminated = False
        reward = 0
        program = self._decode(observation)
        for i, st in enumerate(self.initial_states):
            candidate_program = append_return(program)
            # candidate_program = program
            output = execute_candidate_program(program_string=candidate_program, program_input=st)
            if output == "Invalid Input": 
                terminated = True # TODO: change this back to false

            if "O" in program:
               terminated = True

            if output == self.goal_states[i]:
                reward +=1


        if reward == len(self.initial_states):
            reward = 1.0
        else:
            reward = -1.0
            
        terminated = terminated or reward == 1.0 # or len(program.split("\n")) > 10
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