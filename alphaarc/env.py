from alphaarc.policy.environment import execute_candidate_program
from alphaarc.task import Task
from alphaarc.policy.tokenize import tokenize_task, TextEncoder
from transformers import AutoTokenizer

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



"""
Gym like enviroment for ARC DSL task.

Need to think about how I will structure this now.
At some point I will have to construct it into a string.
The question is whether I do that within the model or the enviroment.


"""
class LineLevelArcEnv:
    def __init__(self, task: Task):
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

        self.n_actions = 5 # n lines of code allowed.
        self.state = []

    # action corresponds to a line of code
    def step(self, action):
        self.state.append(action)
        reward = 0
        observation = (self.initial_states, self.goal_states, self.state)
        terminated = False
        
        for i, state in enumerate(self.initial_states):
            program = "\n".join(self.state)
            program = append_return(program)
            output = execute_candidate_program(program_string=program, program_input=state)
            if output == "Invalid Input": 
                terminated = True
                reward -= -1

            if output == self.goal_states[i]:
                reward +=1
                terminated = True
                

        reward /= len(self.initial_states)
        return observation, reward, terminated
    

    def step(self, action, state): 
        task, state = state

        state.append(action)
        observation = (self.task, state)
        terminated = False
        reward = 0

        for i, st in enumerate(self.initial_states):
            program = "\n".join(state)
            program = append_return(program)
            output = execute_candidate_program(program_string=program, program_input=st)
            if output == "Invalid Input": 
                #terminated = True # TODO: change this back to false
                reward -= 0

            if output == self.goal_states[i]:
                reward +=1
                # terminated = True
                

        terminated = (reward ==  len(self.initial_states))
        reward /= len(self.initial_states)
        return observation, reward, terminated
    

    def get_action_space(self):
        return self.n_actions
 
    def reset(self): 
        self.state = []
        return (self.task, self.state)
    


if __name__ == "__main__": 
    task = Task.from_json('data/training/67385a82.json')
    env = LineLevelArcEnv(task)
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
    result = tokenize_task(task, tokenizer, 100, 1024, 1024)
    print(result)