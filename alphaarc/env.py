from alphaarc.policy.environment import execute_candidate_program, check_syntax
from alphaarc.task import Task
from alphaarc.policy.tokenize import tokenize_task, TextEncoder
from transformers import AutoTokenizer
import copy
import numpy as np
import torch
from alphaarc.augment.mutate_grid import valid_grid

"""
quick note, changed it to where a partial program no longer messes up
"""

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


class ExceededTokenBudget(Exception): 
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class BaseEnv:
    def __init__(self):
        pass

    def set_task(self, task): 
        raise NotImplementedError

class LineLevelArcEnv (BaseEnv):
    def __init__(self, tokenizer_path, n_examples, max_task_len, max_state_len, n_actions,  token_budget):
        self.n_examples = n_examples
        self.input_state_max = max_task_len
        self.max_length = max_state_len
        self.n_actions = n_actions
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.new_line_arr = np.array([NEW_LINE_TOKEN_ID])
        
        
        self._reset_token_budget()

        self.token_budget = token_budget
        
    
    def _reset_token_budget(self):
        self.tokens_used= 0

    def _add_and_check_token_budget(self, action):
        if self.tokens_used > self.token_budget:
            raise ExceededTokenBudget("Exceeded token budget!")
        
        self.tokens_used += len(action)


    def is_below_token_budget(self):
        return self.tokens_used < self.token_budget

    def _add_new_line_if_absent(self, action): 
        if NEW_LINE_TOKEN_ID not in action:
            action = np.concatenate((action, self.new_line_arr))
        return action

    def _decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def _encode(self, program): # TODO: technically incorrect to match.
        return self.tokenizer( program, add_special_tokens=False, return_tensors='np')['input_ids'].squeeze()


    def _if_program_returns(self, string): 
        str_arr = string.split("=")
        lhs = str_arr[0]
        return "O" in lhs

    # action = new program tokens
    # state =  previous program tokens 
    # note that we should only not do token accounting when you have already generated the actions (as in the case of search algs results)
    def step(self, action, state, should_do_token_accounting=True): 
        # action = self._add_new_line_if_absent(action)

        if should_do_token_accounting: 
            self._add_and_check_token_budget(action)

        observation = np.concatenate((state, action))
        terminated = False
        reward = 0
        program = self._decode(observation)


        for i, st in enumerate(self.initial_states):
            candidate_program = append_return(program)
            # candidate_program = program
            output = execute_candidate_program(program_string=candidate_program, program_input=st)
            if output == "Invalid Input": 
                terminated = False # TODO: change this back to false

            if self._if_program_returns(program): 
               terminated = True

            if output == self.goal_states[i]:
                reward +=1


        if reward == len(self.initial_states):
            reward = 1.0 
        else:
            reward = -1.0
        

        

        terminated = terminated or reward == 1.0 # or len(program.split("\n")) > 10
        observation = self._encode(program)
        observation = np.concatenate(([0], observation))
        return observation, reward, terminated


    def is_valid_syntax(self, action, state): 
        action = self._add_new_line_if_absent(action)
        observation = np.concatenate((state, action))
        program = self._decode(observation)
        for i, st in enumerate(self.initial_states):
            candidate_program = append_return(program)
            # candidate_program = program
            output = execute_candidate_program(program_string=candidate_program, program_input=st)
            if output == "Invalid Input": 
                return False   

        return True
    

    def is_valid_syntax(self, program): 
        program = self._decode(program)
        candidate_program = append_return(program)
        result = check_syntax(candidate_program)

        for i, st in enumerate(self.initial_states):
            candidate_program = append_return(program)
            output = execute_candidate_program(program_string=candidate_program, program_input=st)

            if output == "Invalid Input": 
                return False
            
            if type(output ) is not tuple: 
                return False


            if not valid_grid(output):
                return False
            
            if type(output) is str and "Error" in output: 
                return False
            
        return result == "Valid Syntax" 
    
    def get_outputs(self, program): 
        outputs = []
        program = self._decode(program)
        for i, st in enumerate(self.initial_states):
            candidate_program = append_return(program)
            output = execute_candidate_program(program_string=candidate_program, program_input=st)
            outputs.append(output)

        return outputs
        
    
    def evaluate_program(self, program, should_token_account=True): 
        terminated = False
        reward = 0
        program = self._decode(program)
        
        print("- - - - ")
        print(program)
        if should_token_account:
            self._add_and_check_token_budget(program)


        lines = program.split("\n")

        for i, st in enumerate(self.initial_states):
            candidate_program = append_return(program)
            # candidate_program = program
            output = execute_candidate_program(program_string=candidate_program, program_input=st)

            if output == "Invalid Input": 
                terminated = False

            if output != "Invalid Input" and self._if_program_returns(lines[-1]) and output != self.goal_states[i]:
                terminated = True
                reward = -1
               
            if output == self.goal_states[i]:
                reward +=1 




        if reward == len(self.initial_states):
            reward = 1.0 
        else:
            reward = -1.0

        if output == "Invalid Input":
            reward = 0.0

        
        
        if len(program) > self.max_length:
            terminated = True

        
        
        
        terminated = terminated or reward == 1.0

        return reward, terminated

    def get_action_space(self):
        return self.n_actions
 

    def reset(self):
        return np.array([0], dtype=np.int64)


    def set_task(self, task: Task): 
        self.task = copy.deepcopy(task)
        self.initial_states = [
                training_example["input"]
                for training_example in task.training_examples[: self.n_examples]
        ]
        self.goal_states = [
                training_example["output"]
                for training_example in task.training_examples[: self.n_examples]
        ]

        self.initial_states.extend([
                training_example["input"]
                for training_example in task.test_examples
        ])
        self.goal_states.extend([
                training_example["output"]
                for training_example in task.test_examples
        ])


        tokenized_task = np.array(tokenize_task(self.task, self.tokenizer, self.n_examples, self.input_state_max, self.max_length)['input_ids'])
        self.task_length = len(tokenized_task)
        pad_length = (self.input_state_max *2 ) - len(tokenized_task)
        self.tokenized_task = np.pad(tokenized_task, (0, pad_length), constant_values=self.tokenizer.pad_token_id)

        self._reset_token_budget()


if __name__ == "__main__": 
    task = Task.from_json('data/training/67385a82.json')

    #     env = LineLevelArcEnv(task)
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
    result = tokenize_task(task, tokenizer, 100, 1024, 1024)
    print(result)