import re
import ast
from typing import List
from alphaarc.augment.program_sampler import ProgramSampler, ProgramSample
from alphaarc.dsl.primitives import PRIMITIVE_CONSTANTS, PRIMITIVE_FUNCTIONS
from alphaarc.augment.genetic import TaskEvolver
import random
from alphaarc.task import Task
from alphaarc.augment.type_inference import get_primitive_function_type
from alphaarc.dsl.arc_types import Arrow
import traceback

class ProgramCompleter:
    def __init__(self, sampler: ProgramSampler):
        self.sampler = sampler  # we only need the mappings, not the RNG


    def suggest_next_functions(self, program_name, partial_program, I):
        pass

    def suggest_next_args(
        self,
        program_name: str,
        partial_program: str,
        I
    ) -> List[str]:
        
        lines = partial_program.rstrip().splitlines()
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

        try:
            body_ast = ast.parse(finished_part)
            ps.type_inferer.infer_type_from_ast(body_ast)
        
            needs_comma = not partial_line.rstrip().endswith(",")
            dummy_line  = partial_line + (" ___)" if needs_comma else " ___)")
            call_node   = ast.parse(dummy_line.strip()).body[0].value

            func_src = ast.unparse(call_node.func)
            supplied = [ast.unparse(arg) for arg in call_node.args]

            if supplied and supplied[-1] == "___":
                supplied = supplied[:-1]

        except (SyntaxError, TypeError) as e:
            raise ValueError("HERE!")
        
        
        func_types = get_primitive_function_type(func_src)
        


        final_candidates = []
        next_arg_pos   = len(supplied)

        try:
            required_type  = func_types.inputs[next_arg_pos]
        except IndexError as e : # case where all args provided
            return []
         
        candidates = ps.sample_term_with_type(
                    term_type=required_type,
                    terms_to_exclude=[],
                    return_all=True,
                )
        final_candidates.extend(candidates)

        return list(set(final_candidates)) 
        


if __name__ == "__main__":
    #genetic = TaskEvolver('./data/', train_file_path='data/training/', resume=True, primitive_functions=PRIMITIVE_FUNCTIONS, primitive_constants=PRIMITIVE_CONSTANTS )
    # genetic.infer_base_types_for_primitive_functions('./data/', True)
    sampler   = ProgramSampler(data_path="./data/")
    completer = ProgramCompleter(sampler)

    prog_text = """def solve_28bf18c6(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    O = hconcat(x3,"""
    task = Task.from_json('./data/training/28bf18c6.json')
    print(task.program_lines)
    input_ = task.training_examples[0]['input']
    #     input_ = [x['input'] for x in task.training_examples]

    print(completer.suggest_next_args("prog", prog_text, I=input_))