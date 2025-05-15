import argparse
from alphaarc.buffers import ReplayBuffer
from alphaarc.configs import load_config, build_curriculum, build_env
from alphaarc.utils import load_key_split, relabel_task
from alphaarc.env import BaseEnv
from alphaarc.policy.tokenize import tokenize_task
import numpy as np
from tqdm import tqdm
import torch 

from transformers import T5ForConditionalGeneration, AutoTokenizer

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

"""
Algorithm outline: 
-> collect a handful of samples on a particular task
-> anything which is a valid task, add it to the replay buffer 
-> after i collected samples 
"""

def encode_task(task, tokenizer, model, input_state_max=512, n_examples=10, max_length=512): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task



def evaluate_solutions(answers, task, env: BaseEnv, replay_buffer: ReplayBuffer, tokenizer, model):
        for i in range(answers.shape[0]):
            program = answers[i]
            reward, terminated = env.evaluate_program(program, should_token_account=False)
            if env.is_valid_syntax(program):
                new_task = relabel_task(task, env,program)
                new_task = encode_task(task, tokenizer, model)
                program = program.detach().cpu().numpy()
                replay_buffer.add_program_and_task(new_task, program)
            if reward == 1:
                return True



def generate_answers(model, tokenized_task, max_new_length=512, num_return_sequences=1 ):
    answers = model.generate(   tokenized_task.unsqueeze(0),
                                max_new_tokens= max_new_length, 
                                num_return_sequences=num_return_sequences,
                                do_sample=True)

    answers = answers.squeeze(0)
    return answers


def try_solve_task(task, env, replay_buffer,  tokenizer, model): 
    env.set_task(task)
    tokenized_task = torch.tensor(encode_task(task, tokenizer, model)).to('cuda')
    answers = generate_answers(model, tokenized_task) # generate a fixed number of samples, will worry about other stuff later
    solved = evaluate_solutions(answers, task, env, replay_buffer, tokenizer, model)
    return solved


"""
will perform grpo training! 
    -> do the following: 
    -> basically rather than storing the replay buffer, its more like storing a task buffer. So I change it to that. 
    -> This is because I still need to generate new tasks basically.
    -> 
"""
def grpo_train(model, task_buffer):
    for task in task_buffer: 
        pass

        

    

def run_experiment(n_meta_epochs, 
                   curriculum,
                   env,
                   replay_buffer, 
                   tokenizer,
                   model):
     
    solved_task_ids = []
    full_curriculum = curriculum.generate_curriculum()
    for epoch in range(n_meta_epochs):
        for i in tqdm(range(len(full_curriculum))):
            task = full_curriculum[i]
            if try_solve_task(task, env, replay_buffer, tokenizer, model):
                solved_task_ids.extend(task.task_key)
        
        grpo_train()
            
        
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='alphaarc/configs/grpo_config.yaml')
    args = parser.parse_args()
        
    config = load_config(args.config_path)

    replay_buffer = ReplayBuffer()
    curriculum = build_curriculum(config['training_curriculum_config'])
    config = load_config(args.config_path)
    task_key_split = load_key_split('data/split_keys.json')
    curriculum.prune_tasks_not_in_list(tasks_to_keep=task_key_split['val'])
    env = build_env(config['env_config'])
    
    model = T5ForConditionalGeneration.from_pretrained(config['model_path']).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

    run_experiment(n_meta_epochs=1,
                   curriculum=curriculum,
                   env=env,
                   replay_buffer=replay_buffer,
                   tokenizer=tokenizer,
                   model=model)







if __name__ == "__main__": 
    main()