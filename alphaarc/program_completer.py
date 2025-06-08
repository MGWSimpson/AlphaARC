import re
import ast
from typing import List
from alphaarc.augment.program_sampler import ProgramSampler, ProgramSample
from alphaarc.augment.type_inference import display_type
from alphaarc.dsl.primitives import PRIMITIVE_CONSTANTS, PRIMITIVE_FUNCTIONS
from alphaarc.augment.genetic import TaskEvolver
import random
from alphaarc.task import Task
from alphaarc.augment.type_inference import get_primitive_function_type
from alphaarc.dsl.arc_types import Arrow
import traceback

import copy


def get_rhs(line):
    sides = line.split("=")
    return sides[1]

def starts_with_space(s):
    return s.startswith(' ')


def ends_with_space(s):
    return s.endswith(" ")

def find_continuations(input_str, string_list):
    return [s for s in string_list if s.startswith(input_str)]


def is_final_arg(arrow_object, next_arg_pos):
    return len(arrow_object.inputs) == next_arg_pos + 1

def if_program_returns(string): 
        str_arr = string.split("=")
        lhs = str_arr[0]
        return "O" in lhs

class ProgramCompleter:
    def __init__(self, sampler: ProgramSampler):
        self.sampler = sampler  # we only need the mappings, not the RNG
        self.use_sample = None

        
    def _start_program(self):
        return ["O =", f"x1 ="]
    
    # return all possible completions
    def _complete_lhs(self, partial_line, lines): 
        if len(lines) > 2:
            last_line = lines[-2]
            var_name = last_line.split('=')[0].strip()  # "x1"
            
            try: 
                number = int(var_name[1:])
                last_var = number + 1
            except ValueError:
                last_var = 1

        else:
            last_var = 1


        if any(["O" in line.split("=")[0] for line in lines]):
            return []

        to_add = ["O =", f"x{last_var} ="]
        # filter based on partial completions
        filtered = [item for item in to_add if item.startswith(partial_line)]

        return filtered
    
    def _complete_missing_functions(self, partial_line): 
        # we know there is a equals
        rhs = get_rhs(partial_line)
        continuations = find_continuations(rhs.strip(), PRIMITIVE_FUNCTIONS)
        if not starts_with_space(rhs): 
            continuations = [" " + cont for cont in continuations]
        continuations = [cont + "(" for cont in continuations]
        return continuations


    # TODO: need to check for partial args
    def _complete_missing_args(self, partial_program, I, program_name="yes"):
        lines = partial_program.splitlines()
        partial_line = lines[-1]
        finished_part   = "\n".join(lines[:-1])

        if len(lines) == 2:
            finished_part = f"def {program_name}(I):\n    pass"


        sampler = copy.deepcopy(self.sampler)
        ps = ProgramSample(
            program_name              = program_name,
            I                         = I,
            primitive_function_to_general_type_mapping = sampler.primitive_function_to_general_type_mapping,
            primitive_constant_to_type_mapping          =  sampler.primitive_constant_to_type_mapping,
            general_type_to_primitive_function_mapping =  sampler.general_type_to_primitive_function_mapping,
            base_type_to_primitive_function_mapping    = sampler.base_type_to_primitive_function_mapping,
            type_to_primitive_constant_mapping         =  sampler.type_to_primitive_constant_mapping,
            primitive_function_to_base_type_mapping    =  sampler.primitive_function_to_base_type_mapping,
        )

        

        if ")" in partial_line:
            return []

        required_parenth = False
        # check to see if the number of args exceeds

        body_ast = ast.parse(finished_part)
        ps.type_inferer.infer_type_from_ast(body_ast) # parse the valid args and stuff
        


        # this is where I should make that check.
        # if it doesnt start with a ( or a ,

        base = partial_line.rstrip()
        partial_arg = None
        if not base.endswith(",") and not base.endswith("("):
            # Find last ',' or '('
            last_comma = base.rfind(",")
            last_paren = base.rfind("(")
            last_split = max(last_comma, last_paren)

            # If neither found, start from beginning
            if last_split != -1:
                rolled_back = base[:last_split + 1]
                partial_arg = base[last_split + 1:]
            else:
                rolled_back = base + "("
                required_parenth = True

            base = rolled_back

        needs_comma = not base.rstrip().endswith(",")
        dummy_line  = base + (" ___)" if needs_comma else " ___)") # basically check for if its the first arg or not


        call_node   = ast.parse(dummy_line.strip()).body[0].value
        func_src = ast.unparse(call_node.func)
        supplied = [ast.unparse(arg) for arg in call_node.args]

        if supplied and supplied[-1] == "___":
            supplied.pop()


        
        func_types = get_primitive_function_type(func_src)
        
        if func_src in ps.primitive_function_to_base_type_mapping:
            func_types = ps.primitive_function_to_base_type_mapping[func_src]
        else:
            func_types = [get_primitive_function_type(func_src)]
            
            
        final_candidates = []
        next_arg_pos   = len(supplied)


        for x in func_types: 
            required_type  = x.inputs[next_arg_pos]
            try: 
                candidates = ps.sample_term_with_type(
                            term_type=required_type,
                            terms_to_exclude=[],
                            return_all=True,
                        )
                final_candidates.extend(candidates)
            except ValueError: # for some reason it throws an error if it cant find the correct type
                pass
        

        # add in some additional checks here.
        answers = list(set(final_candidates))

        if partial_arg is not None:
            answers = [c for c in answers if c.startswith(partial_arg)]





        if required_parenth:
            answers = ["("+ ans for ans in answers]
        elif not ends_with_space(partial_line) and not partial_line.endswith("("):
            answers = [" "+ ans for ans in answers]
        
        # answers = partial line

        if is_final_arg(x, next_arg_pos): 
            return [ans + (")" if if_program_returns(ans) else ")\n") for ans in answers]
        else: 
            return [ans +"," for ans in answers]


    def _complete_rhs(self, partial_line, lines, program_text, sample_task_input): 
        
        # need to figure out if you are completing a function or an argument.
        if all(prim_fun not in partial_line for prim_fun in PRIMITIVE_FUNCTIONS): # in the case we are missing a function. Lets return all functions
            return self._complete_missing_functions(partial_line)
        else:
            return self._complete_missing_args(program_text, sample_task_input)
        

    def complete(self,program_text, sample_task_input, program_name="EMPTY"): 
        
        lines = program_text.split("\n")
        lines = [l.strip() for l in lines]
        
        if len(lines) > 50:
            return []


        if len(lines) == 1:
            return self._start_program()
        
        partial_line = lines[-1]

        if "=" not in partial_line:
            return self._complete_lhs(partial_line , lines)
        else:
            return self._complete_rhs(partial_line, lines, program_text, sample_task_input)


def format_as_dummy_program(program_lines):
    return f"""def solve_28bf18c6(I):
    {program_lines}"""


if __name__ == "__main__":
    #genetic = TaskEvolver('./data/', train_file_path='data/training/', resume=True, primitive_functions=PRIMITIVE_FUNCTIONS, primitive_constants=PRIMITIVE_CONSTANTS )
    # genetic.infer_base_types_for_primitive_functions('./data/', True)
    sampler   = ProgramSampler(data_path="./data/")
    completer = ProgramCompleter(sampler)


    #    prog_text = format_as_dummy_program(prog_text)
    #print(prog_text)
    prog_text = """"""
    task = Task.from_json('./data/training/28bf18c6.json')
    print(task.program)
    input_ = task.training_examples[0]['input']

    print(completer.complete(prog_text, input_))
