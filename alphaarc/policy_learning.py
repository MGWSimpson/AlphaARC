"""
Class which contains all the code for the chapter on learning from mistakes.
"""


import argparse
from alphaarc.buffers import ReplayBuffer
from alphaarc.configs import load_config, build_curriculum, build_env
from alphaarc.utils import load_key_split, relabel_task
from alphaarc.env import BaseEnv
import numpy as np
from tqdm import tqdm
from grpo import GRPOTrainer
import torch 
import pytorch_lightning as pl
from collections import defaultdict
import json
from transformers import T5ForConditionalGeneration, AutoTokenizer
import wandb
import random
import os
import shutil
from alphaarc.policy.tokenize import tokenize_task

from alphaarc.utils import save_answer, prepare_output_dir, save_stats_to_file, save_model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


# --helpers --


def encode_task(task, tokenizer, model, input_state_max=256, n_examples=10, max_length=256): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task

# -- end helpers --

def evaluate_solutions(answers, task, env: BaseEnv, relabelled_tasks, tokenizer, model, answer_dict):
        for i in range(answers.shape[0]):
            program = answers[i]
            reward, terminated = env.evaluate_program(program, should_token_account=False)
            
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


def try_solve_task(task, env, relabelled_tasks,  tokenizer, model, answer_dict, grpo_trainer=None): 
    env.set_task(task)

    if grpo_trainer is None:
        tokenized_task = torch.tensor(encode_task(task, tokenizer, model)).to('cuda')
        answers = generate_answers(model, tokenized_task) # generate a fixed number of samples, will worry about other stuff later
    
    else:
        answers = grpo_trainer.generate_answers([task], env,  24)

    
    solved = evaluate_solutions(answers, task, env, relabelled_tasks
                                , tokenizer, model, answer_dict)
    return solved




def run_experiment(n_meta_epochs, 
                   curriculum,
                   env,
                   replay_buffer, 
                   tokenizer,
                   model,
                   grpo_trainer: GRPOTrainer,
                   save_model_every=10,
                   output_dir='./results/method'):
     
    solved_task_ids = []
    full_curriculum = curriculum.generate_curriculum()
    full_curriculum = full_curriculum
    answers_dict = defaultdict(list)
    epoch_stats = []

    for epoch in tqdm(range(n_meta_epochs)):
        random.shuffle(full_curriculum)
        solved_this_epoch = []
        for i in tqdm(range(len(full_curriculum))):
            task = full_curriculum[i]
            relabelled_tasks = []
            if try_solve_task(task, env, relabelled_tasks, tokenizer, model, answers_dict, grpo_trainer):
                solved_this_epoch.append(task.task_key)
                print(f"solved: {len(set(solved_task_ids))}")

        solved_task_ids.extend(solved_this_epoch)
        if epoch % save_model_every == 0:
            save_model(model, output_dir, epoch)

        epoch_stats.append({
            "epoch": epoch,
            "solved_this_epoch": len(solved_this_epoch),
            "cumulative_solved": len(set(solved_task_ids)),
        })
        save_stats_to_file(epoch_stats, output_dir)

            
        
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='alphaarc/configs/policy_learning/grpo.yaml')
    args = parser.parse_args()
        
    config = load_config(args.config_path)

    replay_buffer = ReplayBuffer()
    curriculum = build_curriculum(config['training_curriculum_config'])
    config = load_config(args.config_path)
    
    task_key_split = load_key_split('data/split_keys.json')
    curriculum.prune_tasks_not_in_list(tasks_to_keep=task_key_split['val'])
    env = build_env(config['env_config'])
    
    model = T5ForConditionalGeneration.from_pretrained(config['model_path']).to('cuda')
    ref_model = T5ForConditionalGeneration.from_pretrained(config['model_path']).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])



    method = config['method']

    if method == "GRPO": 
        grpo_trainer = GRPOTrainer( ref_model, 
                                    model,
                                    tokenizer,
                                    env)
    elif method == "SPARSEGRPO": 
        grpo_trainer = GRPOTrainer(ref_model,
                                   model,
                                   tokenizer, 
                                   sparse_variant=True)    
    
    elif method == "INTERNALGRPO": 
        grpo_trainer = GRPOTrainer(ref_model,
                                   model,
                                   tokenizer, 
                                   sparse_variant=True)    
    elif method == "SAMPLE":
        grpo_trainer = None 
    else:
        raise ValueError('Specified method does not exist')
    
    
    output_dir =  f"results/{method.lower()}"
    prepare_output_dir(output_dir)

    pl.seed_everything(0)
    run_experiment(n_meta_epochs=10,
                   curriculum=curriculum,
                   env=env,
                   replay_buffer=replay_buffer,
                   tokenizer=tokenizer,
                   model=model,
                   grpo_trainer=grpo_trainer,
                   output_dir=output_dir)







if __name__ == "__main__": 
    main()