import argparse
from alphaarc.buffers import ReplayBuffer
from alphaarc.configs import load_config, build_curriculum, build_env
from alphaarc.utils import load_key_split, relabel_task
from alphaarc.env import BaseEnv
from alphaarc.policy.tokenize import tokenize_task
import numpy as np
from tqdm import tqdm
from grpo import GRPOTrainer
import torch 
import pytorch_lightning as pl
from collections import defaultdict
import json
from transformers import T5ForConditionalGeneration, AutoTokenizer

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

"""
Algorithm outline: 
-> collect a handful of samples on a particular task
-> anything which is a valid task, add it to the replay buffer 
-> after i collected samples 

-> likely something I can do where it retrains on successes too....
"""

def encode_task(task, tokenizer, model, input_state_max=512, n_examples=10, max_length=512): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task



def save_answer(answer_dict):
    with open("data.jsonl", "w") as f:
        json.dump(answer_dict, f)

def evaluate_solutions(answers, task, env: BaseEnv, relabelled_tasks, tokenizer, model, answer_dict):
        for i in range(answers.shape[0]):
            program = answers[i]
            reward, terminated = env.evaluate_program(program, should_token_account=False)
            if env.is_valid_syntax(program):
                new_task = relabel_task(task, env, program, tokenizer.decode(program, skip_special_tokens=True, clean_up_tokenization_spaces=True))
                relabelled_tasks.append(new_task)

            if reward == 1:
                answer_dict[task.task_key].append(tokenizer.decode(program, skip_special_tokens=True, clean_up_tokenization_spaces=True))
                save_answer(answer_dict)
                return True

        return False


def generate_answers(model, tokenized_task, max_new_length=512, num_return_sequences=24 ):
    answers = model.generate(   tokenized_task.unsqueeze(0),
                                max_new_tokens= max_new_length,
                                num_return_sequences=num_return_sequences,
                                do_sample=True)

    answers = answers.squeeze(0)
    return answers


def try_solve_task(task, env, relabelled_tasks,  tokenizer, model, answer_dict): 
    env.set_task(task)
    tokenized_task = torch.tensor(encode_task(task, tokenizer, model)).to('cuda')
    answers = generate_answers(model, tokenized_task) # generate a fixed number of samples, will worry about other stuff later
    solved = evaluate_solutions(answers, task, env, relabelled_tasks
                                , tokenizer, model, answer_dict)
    return solved



   
"""
So this is where im at. basically i need to get a list of the relabelled tasks rather than the buffer
""" 
def run_experiment(n_meta_epochs, 
                   curriculum,
                   env,
                   replay_buffer, 
                   tokenizer,
                   model,
                   grpo_trainer: GRPOTrainer):
     
    solved_task_ids = []
    full_curriculum = curriculum.generate_curriculum()
    full_curriculum = full_curriculum
    relabelled_tasks = [] 

    answers_dict = defaultdict(list)

    for epoch in tqdm(range(n_meta_epochs)):
        for i in tqdm(range(len(full_curriculum))):
            task = full_curriculum[i]
            relabelled_tasks = []
            if try_solve_task(task, env, relabelled_tasks, tokenizer, model, answers_dict):
                solved_task_ids.append(task.task_key)
                print(f"solved: {len(set(solved_task_ids))}")
                print(set(solved_task_ids))

            grpo_trainer.train(relabelled_tasks)

            
        
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


    grpo_trainer = GRPOTrainer(T5ForConditionalGeneration.from_pretrained(config['model_path']).to('cuda'), 
                               model,
                               tokenizer,
                               env)

    pl.seed_everything(0)
    run_experiment(n_meta_epochs=100,
                   curriculum=curriculum,
                   env=env,
                   replay_buffer=replay_buffer,
                   tokenizer=tokenizer,
                   model=model,
                   grpo_trainer=grpo_trainer)







if __name__ == "__main__": 
    main()