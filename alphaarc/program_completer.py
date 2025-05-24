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

class ProgramCompleter:
    def __init__(self, sampler: ProgramSampler):
        self.sampler = sampler  # we only need the mappings, not the RNG

    def _start_program(self):
        return ["O =", f"x1 ="]
    
    # return all possible completions
    def _complete_lhs(self, partial_line, lines): 
        if len(lines) >= 2:
            last_line = lines[-2]
            var_name = last_line.split('=')[0].strip()  # "x1"
            number = int(var_name[1:])
            last_var = number + 1
        else:
            last_var = 1

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

        ps = ProgramSample(
            program_name              = program_name,
            I                         = I,
            primitive_function_to_general_type_mapping = self.sampler.primitive_function_to_general_type_mapping,
            primitive_constant_to_type_mapping          = self.sampler.primitive_constant_to_type_mapping,
            general_type_to_primitive_function_mapping = self.sampler.general_type_to_primitive_function_mapping,
            base_type_to_primitive_function_mapping    = self.sampler.base_type_to_primitive_function_mapping,
            type_to_primitive_constant_mapping         = self.sampler.type_to_primitive_constant_mapping,
            primitive_function_to_base_type_mapping    = self.sampler.primitive_function_to_base_type_mapping,
        )

        body_ast = ast.parse(finished_part)
        ps.type_inferer.infer_type_from_ast(body_ast)
        
        needs_comma = not partial_line.rstrip().endswith(",")
        dummy_line  = partial_line + (" ___)" if needs_comma else " ___)")
        call_node   = ast.parse(dummy_line.strip()).body[0].value
        func_src = ast.unparse(call_node.func)
        supplied = [ast.unparse(arg) for arg in call_node.args]

        if supplied and supplied[-1] == "___":
            supplied = supplied[:-1]

        
        
        func_types = get_primitive_function_type(func_src)
        
        if func_src in ps.primitive_function_to_base_type_mapping:
            func_types = ps.primitive_function_to_base_type_mapping[func_src]
        else:
            print("uh oh")
            func_types = ps.type_inferer.type_dict[func_src][0]   # variable Arrow


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
            except ValueError: # for some reason it throws an error if n
                pass
        

        # add in some additional checks here.
        
        answers = list(set(final_candidates))



        # check for a space in the last bit of the partial line

        print(ends_with_space(partial_line))
        if not ends_with_space(partial_line):
            answers = [" "+ ans for ans in answers]


        if is_final_arg(x, next_arg_pos): 
            return [ans +")\n" for ans in answers]
        else: 
            return [ans +"," for ans in answers]


    def _complete_rhs(self, partial_line, lines, program_text, sample_task_input): 
        
        # need to figure out if you are completing a function or an argument.
        if all(prim_fun not in partial_line for prim_fun in PRIMITIVE_FUNCTIONS): # in the case we are missing a function. Lets return all functions
            return self._complete_missing_functions(partial_line)
        else:
            return self._complete_missing_args(program_text, sample_task_input)
        

    def complete(self,program_text, sample_task_input, program_name="EMPTY"): 
        
        lines = program_text.splitlines()
        lines = [l.strip() for l in lines]
        
        if len(lines) == 0:
            return self._start_program()
        
        partial_line = lines[-1]


        if "=" not in partial_line:
            return self._complete_lhs(partial_line , lines)
        else:
            return self._complete_rhs(partial_line, lines, program_text, sample_task_input)

if __name__ == "__main__":
    #genetic = TaskEvolver('./data/', train_file_path='data/training/', resume=True, primitive_functions=PRIMITIVE_FUNCTIONS, primitive_constants=PRIMITIVE_CONSTANTS )
    # genetic.infer_base_types_for_primitive_functions('./data/', True)
    sampler   = ProgramSampler(data_path="./data/")
    completer = ProgramCompleter(sampler)

    prog_text = """def solve_28bf18c6(I):
    x1 = objects(I, T,"""
    task = Task.from_json('./data/training/28bf18c6.json')
    print(task.program)
    input_ = task.training_examples[0]['input']

    print(completer.complete(prog_text, input_))
