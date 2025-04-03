from alphaarc.policy.environment import execute_candidate_program
from alphaarc.task import Task


def append_action_to_state(state, action): 
    return state + action


# appends a return statement if it does not return anything
def append_return(program):
    lines = program.split("\n")
    if "return O" not in lines[-1]:
        program += f"\nO = {lines[-2].split("=")[0]}\n return O"
    
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

        self.state = ""

    # action corresponds to a line of code
    def step(self, action):
        self.state += action
        reward = 0
        observation = (self.initial_states, self.goal_states, self.state)
        terminated = False
        
        for i, state in enumerate(self.initial_states):
            program = self.state
            program = append_return(program)
            output = execute_candidate_program(program_string=program, program_input=state)
            if output == "Invalid Input": 
                terminated = True
                reward -= -1
                break

            if output == self.goal_states[i]:
                reward +=1
                break

        reward /= len(self.initial_states)
        return observation, reward, terminated
 

 
    def reset(self): 
        self.state = ""
        return (self.initial_states, self.goal_states, self.state)
    


if __name__ == "__main__": 
    #

    env = 