import re 
import os
import copy
from typing import Dict
from alphaarc.task import Task
from alphaarc import PROJECT_FOLDER_PATH
from alphaarc.policy.environment import execute_candidate_program
from pathlib import Path
import json
from pathlib import Path
import json
from queue import Queue

def find_grid_functions(filename):
    with open(filename, "r") as file:
        contents = file.read()
    pattern = re.compile(
    r'^\s*def\s+(\w+)'           # capture the function name
    r'(?=\s*\([^)]*\)\s*'        # look-ahead to be sure it is followed by …
    r'->\s*Grid\s*:)',           # … “-> Grid:”
    flags=re.MULTILINE | re.DOTALL)
    function_names = re.findall(pattern, contents)
    return function_names




GRID_FUNCTIONS = find_grid_functions(
    filename=os.path.join(PROJECT_FOLDER_PATH, "alphaarc/dsl/dsl.py")
)
STANDALONE_I = re.compile(r'\bI\b')      

# TODO: could potentially be made more complete with the check for Any types
# TODO: I could actually double down on this further by getting any intermediate grid and setting that as the output.
def contains_grid_type(line):
    for grid_func in GRID_FUNCTIONS:
        if grid_func in line:
            return True
    return False


def get_lines_vars(line_array):
    return [line.split("=")[0].strip() for line in line_array]


def check_any_var_used(vars, program_lines): 
    for line in program_lines:
        for var in vars:
            if var in line:
                return True

    return False


def check_input_used(program_lines):
    for line in program_lines: 
        if bool(STANDALONE_I.search(line)):
            return True
    return False


def can_be_compressed(line, line_index, program_lines):
    previous_lines = program_lines[:line_index]
    future_lines = program_lines[line_index +1:]
    previous_lines_vars = get_lines_vars(previous_lines)
    return (not check_any_var_used(previous_lines_vars, future_lines)) and (not check_input_used(future_lines))



def convert_line_to_output_line(line): 
    lhs, rhs = line.split('=', 1)          # split once, just at the first '='
    return f"O ={rhs}" 


# may somewhat break with just a single output but can cope with that when we get there (or just not transform those)
def create_left_program_string(program_lines):
    program_lines[-1] = convert_line_to_output_line(program_lines[-1])
    program_string = "\n".join(program_lines)
    return program_string


def rename_lhs_vars(lines):
    
    lhs_to_new: Dict[str, str] = {}
    counter = 1
    for line in lines:
        m = re.match(r'\s*([A-Za-z_]\w*)\s*=', line)
        if m:
            name = m.group(1)
            if name not in lhs_to_new:
                lhs_to_new[name] = f"x{counter}"
                counter += 1

    # Nothing to rename?  Just hand the original back.
    if not lhs_to_new:
        return lines

    # Pass 2 ── swap every whole-word hit with its new name
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, lhs_to_new)) + r')\b')

    def repl(match):
        return lhs_to_new[match.group(0)]

    return [pattern.sub(repl, line) for line in lines]

def create_right_program_string(program_lines, pruned_variable_name): 
    if len(program_lines) == 1:
        program_lines[0] = program_lines[0].replace(pruned_variable_name, "I")
        return program_lines[0]

    program_lines = program_lines[1:]
    

    for i, line in enumerate(program_lines):

        new_line = line.replace(pruned_variable_name, "I")
        program_lines[i] = new_line
    
    
    
    lines = rename_lhs_vars(program_lines)

    lines[-1] = convert_line_to_output_line( lines[-1])
    
    return "\n".join(lines)
    
    
    


def create_task_dict(inputs, outputs):
    return {"input": inputs, "output": outputs}

def create_new_tasks(line, line_index, program_lines, task): 
    
    new_training_targets = []
    new_test_targets = []
    left_program_string = create_left_program_string(program_lines[:line_index + 1])
    
    for inp in [x['input'] for x in task.training_examples]:
        output = execute_candidate_program( left_program_string, inp)
        new_training_targets.append(output)

    for inp in [x['input'] for x in task.test_examples]:
        output = execute_candidate_program( left_program_string, inp)
        new_test_targets.append(output)



    left_training_examples = []
    left_test_examples = []

    for i, inp in enumerate([x['input'] for x in task.training_examples]):
        left_training_examples.append(create_task_dict(inp, new_training_targets[i]))

    for i, inp in enumerate([x['input'] for x in task.test_examples]):
        left_test_examples.append(create_task_dict(inp, new_test_targets[i]))
    

    left_task = Task(left_program_string, left_training_examples, left_test_examples, task.task_key+ str(line_index) + "L", parent_key=task.task_key )

    pruned_var = get_lines_vars([line])[0]
    right_program_string = create_right_program_string(program_lines[line_index:], pruned_variable_name=pruned_var)


    right_training_example = []
    right_test_example = []


    for i, inp in enumerate([x['output'] for x in task.training_examples]):
        right_training_example.append(create_task_dict(new_training_targets[i], inp))

    for i, inp in enumerate([x['output'] for x in task.test_examples]):
        right_test_example.append(create_task_dict(new_test_targets[i], inp))
    

    right_task = Task(right_program_string, right_training_example, right_test_example, task.task_key+ str(line_index) + "R", parent_key=task.task_key )



    

    return [left_task, right_task]
    
    

    

def compress(program_lines: list, task: Task):
    new_tasks = []

    for i, line in enumerate(program_lines): 
        if contains_grid_type(line) and can_be_compressed(line,i, program_lines): 
            new_tasks.extend(create_new_tasks(line, i, program_lines, task))


    return new_tasks

def collate_task_inputs(task: Task): 
    program_inputs = []

    train_inputs = [x['input'] for x in task.training_examples]
    test_input = [x['input'] for x in task.test_examples]
    
    program_inputs.extend(train_inputs)
    program_inputs.extend(test_input)

    return program_inputs



"""
Need to add this thing where basically any new tasks are then added back to the queue

"""
def main():
    data_dir = Path("data/training")    # directory with your JSON files
    task_queue = Queue()              # queue for new tasks
    processed_tasks = []    

    for json_file in data_dir.glob("*.json"):     # iterate over every .json in the folder
        task = Task.from_json(json_file)          # load each file
        task_queue.put(task)


    while not task_queue.empty():
        task = task_queue.get()
        program_lines = task.program_lines.split("\n")
        compressed_tasks = compress(program_lines, task)
        for new_task in compressed_tasks:
            task_queue.put(new_task)
            processed_tasks.append(new_task)

    json_objects = [task.to_dict() for task in processed_tasks]

    with open("data/new_tasks.jsonl", "w") as f:
        for item in json_objects:
            f.write(json.dumps(item) + "\n")


    

    




if __name__ == "__main__": 
    main()
