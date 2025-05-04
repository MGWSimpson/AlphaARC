import os
import argparse
from alphaarc.configs import load_config, build_curriculum
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import numpy as np 
from alphaarc.policy.tokenize import tokenize_task




os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
NEW_LINE_TOKEN_ID = 203

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


        tokenized_tasks.append(tokenized_task)
        attention_masks.append(attention_mask)

    return tokenized_tasks, attention_masks


class SampleAndFilterSolver:
    def __init__(self, 
                 model_path, 
                 tokenizer_path, 
                 max_new_length,
                 batch_size):
        
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)



    def _generate_answers(): 
        pass

    # returns solve rate things like that 
    def solve_tasks(self, tasks: list):
        tokenized_tasks, attention_masks = tokenize_task_arr(tasks, self.tokenizer)
        
        answers = self._generate_answers

def run_experiment(n_epochs, batch_size, solver: SampleAndFilterSolver, curriculum): 
    for meta_epoch in tqdm(range(n_epochs)):
        full_curriculum = curriculum.generate_curriculum()

        for i in tqdm(range(0, len (full_curriculum), batch_size)):
            batch = full_curriculum[i:i+batch_size]
            results = solver.solve_tasks(batch)



# collect arguments and run experiment
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='alphaarc/configs/sample_filter_config.yaml')
    args = parser.parse_args()

    config = load_config(args.config_path)

    solver = SampleAndFilterSolver(config['model_path'], config['tokenizer_path'], config['max_new_length'], config['batch_size'])

    curriculum = build_curriculum(config['training_curriculum_config'])
    
    run_experiment(config['n_epochs'], config['batch_size'], solver, curriculum )

if __name__ == "__main__":
    main()



