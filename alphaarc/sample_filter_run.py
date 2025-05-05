import os
import argparse
from alphaarc.configs import load_config, build_curriculum, build_env
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import numpy as np 
from alphaarc.policy.tokenize import tokenize_task
import torch 
from alphaarc.env import BaseEnv, ExceededTokenBudget

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
NEW_LINE_TOKEN_ID = 203


"""
NOTE:
-> Imposing an assumption that only one task is being solved at a time. This makes tracking the tasks much easier
"""

"""
TODO: 
-> Will need to remove it from curriculum if it solves. Will do this later. 
-> saving which tasks it solves
"""
def tokenize_task_arr(task_arr, tokenizer, input_state_max=512, n_examples=10, max_length=512): 
    tokenized_tasks = []
    attention_masks = []
    for task in task_arr:
        tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max-1, max_length)['input_ids'])
        tokenized_task = np.concatenate((tokenized_task,  np.array([NEW_LINE_TOKEN_ID])))
        
        attention_mask = np.ones(tokenized_task.shape)
        
        pad_length = (input_state_max *2 ) - len(tokenized_task)
        
        tokenized_task = np.pad(tokenized_task, (0, pad_length), constant_values=tokenizer.pad_token_id)
        attention_mask = np.pad(attention_mask, (0, pad_length), constant_values=0)


        tokenized_task = torch.tensor(tokenized_task)
        attention_mask = torch.tensor(attention_mask)

        tokenized_tasks.append(tokenized_task)
        attention_masks.append(attention_mask)

    return tokenized_tasks, attention_masks



def evaluate_solutions(answers, task, env: BaseEnv):
    
    env.set_task(task)

    solved = False

    for i in range(answers.shape[0]):
        program = answers[i]
        reward, terminated = env.evaluate_program(program)
        if reward == 1:
            solved = True
        

    return solved


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

    def _generate_answers(self, tokenized_tasks, attention_masks): 
        answers = self.model.generate(  input_ids=tokenized_tasks,
                                        attention_mask=attention_masks,
                                        max_new_tokens= self.max_new_length,
                                        num_return_sequences=self.num_return_sequences,
                                        do_sample=True)

        answers = answers.view(self.batch_size, self.num_return_sequences, -1)
        return answers 


    def solve_tasks(self, tasks: list, env):
        try:
            while True:
                tokenized_tasks, attention_masks = tokenize_task_arr(tasks, self.tokenizer)
                tokenized_tasks, attention_masks = torch.stack(tokenized_tasks) , torch.stack(attention_masks)
                tokenized_tasks, attention_masks = tokenized_tasks.to(self.device), attention_masks.to(self.device)
                answers = self._generate_answers(tokenized_tasks, attention_masks)

                for i in range(len(tasks)):
                    solved = evaluate_solutions(answers[i], tasks[i], env)
                    if solved:
                        return [tasks[i].task_key] 
                    
        except ExceededTokenBudget:
            print("exceeded token budget!")
            return []
            
def run_experiment(n_epochs, batch_size, solver: SampleAndFilterSolver, curriculum, env): 
    solved_task_ids = []

    for meta_epoch in tqdm(range(n_epochs)):
        full_curriculum = curriculum.generate_curriculum()

        for i in tqdm(range(0, len (full_curriculum), batch_size)):
            batch = full_curriculum[i:i+batch_size]
            solved_task_ids.extend( solver.solve_tasks(batch, env))
            print(len(solved_task_ids))
            print(solved_task_ids)

       

# collect arguments and run experiment
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='alphaarc/configs/sample_filter_config.yaml')
    args = parser.parse_args()

    config = load_config(args.config_path)

    solver = SampleAndFilterSolver(config['model_path'], config['tokenizer_path'], config['max_new_length'], config['batch_size'], num_return_sequences=config['num_return_sequences'])

    curriculum = build_curriculum(config['training_curriculum_config'])

    env = build_env(config['env_config'])

    print(run_experiment(config['n_epochs'], config['batch_size'], solver, curriculum, env))

if __name__ == "__main__":
    main()



