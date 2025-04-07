from alphaarc.policy.environment import execute_candidate_program
from alphaarc.task import Task


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
        state.append(action)
        observation = (self.initial_states, self.goal_states, state)
        terminated = False
        
        for i, state in enumerate(self.initial_states):
            program = "\n".join(state)
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
    

    def get_action_space(self):
        return self.n_actions
 
    def reset(self): 
        self.state = []
        return (self.initial_states, self.goal_states, self.state)
    


if __name__ == "__main__": 
    task = Task.from_json('data/training/67385a82.json')
    env = LineLevelArcEnv(task)
    program1 = """x1 = objects(I, T, F, F)"""
    program2 = "x2 = colorfilter(x1, THREE)"
    
    print(env.step(program1))  
    print(env.step(program2))
   