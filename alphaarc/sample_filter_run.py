import os
import argparse
from alphaarc.configs import load_config, build_curriculum, build_env
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import numpy as np 
from alphaarc.policy.tokenize import tokenize_task
import torch 
from alphaarc.env import BaseEnv, ExceededTokenBudget
from alphaarc.buffers import ReplayBuffer
from alphaarc.utils import relabel_task
from alphaarc.utils import load_key_split
import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


"""

NOTE: 
-> the prior stuff had like a new line concat for some reason. 

NOTE:
-> Imposing an assumption that only one task is being solved at a time. This makes tracking the tasks much easier
"""

"""
At a budget of 50k tokens, it scored 5/89

['ac0a08a4', 'd9fac9be', '4258a5f9', '6fa7a44f', '6150a2bd']
"""

# TODO: add hindsight relabel.
# 50k tokens per task, but rather than it working on the task thing, we do like 10k then move on etc
        # -> can just implement this with like the meta iterations and reducing the budgets. 
# if its a syntactically correct program, then we relabel it to somethign else.
# need to check each time we are evaluating the programs. are they syntactically correct
# if so then save it, for now however we will just flag it.
def tokenize_task_arr(task_arr, tokenizer, input_state_max=512, n_examples=10, max_length=512): 
    tokenized_tasks = []
    attention_masks = []
    for task in task_arr:
        tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
        
        attention_mask = np.ones(tokenized_task.shape)
        
        pad_length = (input_state_max *2 ) - len(tokenized_task)
        
        tokenized_task = np.pad(tokenized_task, (0, pad_length), constant_values=tokenizer.pad_token_id)
        attention_mask = np.pad(attention_mask, (0, pad_length), constant_values=0)


        tokenized_task = torch.tensor(tokenized_task)
        attention_mask = torch.tensor(attention_mask)

        tokenized_tasks.append(tokenized_task)
        attention_masks.append(attention_mask)

    return tokenized_tasks, attention_masks



def encode_task(task, tokenizer, model, input_state_max=512, n_examples=10, max_length=512): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task





class SampleAndFilterSolver:
    def __init__(self, 
                 model_path, 
                 tokenizer_path, 
                 max_new_length,
                 batch_size,
                 num_return_sequences,
                 device='cuda'):
        
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_new_length = max_new_length
        self.batch_size = batch_size
        self.num_return_sequences = num_return_sequences

    def evaluate_solutions(self, answers, task, env: BaseEnv, replay_buffer: ReplayBuffer):
        for i in range(answers.shape[0]):
            program = answers[i]
            reward, terminated = env.evaluate_program(program)
            
            
            if env.is_valid_syntax(program):
                
                new_task = relabel_task(task, env,program)
                new_task = encode_task(task, self.tokenizer, self.model)
                program = program.detach().cpu().numpy()
                replay_buffer.add_program_and_task(new_task, program)
            if reward == 1:
                return True
                

        return False
    def _generate_answers(self, encoder_outputs): 
        answers = self.model.generate(  encoder_outputs.unsqueeze(0),
                                        max_new_tokens= self.max_new_length, 
                                        num_return_sequences=self.num_return_sequences,
                                        do_sample=True)

        answers = answers.view(self.batch_size, self.num_return_sequences, -1)
        answers = answers.squeeze(0)
        return answers 


    def solve_task(self, task, env, replay_buffer): 
        try:
            encoder_outputs =  torch.tensor(encode_task(task, self.tokenizer, self.model)).to('cuda')
            env.set_task(task)

            while True:
                answers = self._generate_answers(encoder_outputs)
                solved = self.evaluate_solutions(answers, task, env, replay_buffer)
                
                if solved:
                    return [task. task_key] 
                
                print(f"budget left: {env.token_budget - env.tokens_used}")
        except ExceededTokenBudget:
            print("exceeded token budget!")
            return [] 



def train(solver, replay_buffer): 
    pass


def run_experiment(n_epochs, batch_size, solver: SampleAndFilterSolver, curriculum, env, replay_buffer): 
    solved_task_ids = []
    full_curriculum = curriculum.generate_curriculum()
    for epoch in range(n_epochs):
        for i in tqdm(range(0, len (full_curriculum), batch_size)):
            batch = full_curriculum[i:i+batch_size]
            solved_task_ids.extend( solver.solve_task(batch[0], env, replay_buffer))

            # perform some type of replay buffer.
            print(len(solved_task_ids))
            print(solved_task_ids)

        # perform the training here.
        train(solver, replay_buffer)
       

# collect arguments and run experiment
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='alphaarc/configs/sample_filter_config.yaml')
    args = parser.parse_args()
    replay_buffer = ReplayBuffer()


    config = load_config(args.config_path)
    solver = SampleAndFilterSolver(config['model_path'], config['tokenizer_path'], config['max_new_length'], config['batch_size'], num_return_sequences=config['num_return_sequences'])
    curriculum = build_curriculum(config['training_curriculum_config'])
    task_key_split = load_key_split('data/split_keys.json')
    curriculum.prune_tasks_not_in_list(tasks_to_keep=task_key_split['val'])
    env = build_env(config['env_config'])
    print(run_experiment(config['n_epochs'], config['batch_size'], solver, curriculum, env, replay_buffer ))

if __name__ == "__main__":
    main()



