"""
minimal implementation of GRPO. Probably not super efficient but hacked together for my need.

note: for now just doing one task at a time, will batchify later (removes the padding issue :D )

reference implementation: https://github.com/lsdefine/simple_GRPO/
"""
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
import random
from tqdm import tqdm
import torch
from alphaarc.task import Task
import numpy as np
from alphaarc.policy.tokenize import tokenize_task
from alphaarc.env import LineLevelArcEnv, BaseEnv


def encode_task(task, tokenizer, model, input_state_max=512, n_examples=10, max_length=512): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task


class GRPOTrainer:

    def __init__(self,
                 ref_model: T5ForConditionalGeneration,
                 policy_model: T5ForConditionalGeneration,
                 tokenizer: AutoTokenizer,
                 env: BaseEnv,
                 num_gen_per_group=8,
                 batch_size=2, 
                 lr=1e-6,
                 beta= 0.04): 
        
        self.tokenizer = tokenizer
        self.optimizer = AdamW(policy_model.parameters(), lr=lr)
        
        self.ref_model = ref_model
        self.ref_model.requires_grad_(False)
        

        self.policy_model = policy_model
        self.env = env
        

        # grpo params
        self.num_gen_per_group = num_gen_per_group # we will sneak in the correct answer ! -> hindsight stuff
        self.batch_size = batch_size
        self.beta = beta


    """
    Need to change this over to be multiple decoder input ids
    """
    def _compute_reward(self, task, decoder_input_ids): 
        self.env.set_task(task)
        rewards = [self.env.evaluate_program(x, should_token_account=False)[0] for x in decoder_input_ids]
        return torch.tensor(rewards, dtype=torch.float, device='cuda')
        
    def _compute_logits(self, input_ids, decoder_input_ids, model): 
        logits = model(input_ids=input_ids.repeat(self.num_gen_per_group, 1), decoder_input_ids=decoder_input_ids).logits
        return logits

    
    def _compute_per_token_log_probs(self,logits, decoder_input_ids): 
        per_token_log_probs = []
        for logits_row, input_ids_row in zip(logits, decoder_input_ids): 
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_log_probs.append(token_log_prob)

        return torch.stack(per_token_log_probs)


    # returns the loss
    def _grpo_step(self, input_ids, decoder_input_ids, rewards): 
        


        # get the logits :D 
        policy_logits =  self._compute_logits(input_ids, decoder_input_ids, self.policy_model )
        ref_logits = self._compute_logits(input_ids, decoder_input_ids, self.ref_model)



        # compute the log probs
        per_token_log_probs = self._compute_per_token_log_probs(policy_logits, decoder_input_ids)
        ref_per_token_log_probs = self._compute_per_token_log_probs(ref_logits, decoder_input_ids)



        # kl
        per_token_kl = torch.exp(ref_per_token_log_probs - per_token_log_probs) - (ref_per_token_log_probs - per_token_log_probs) - 1

        # handle masking, one thing at a time tho
        #completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

        mean_grouped_rewards = rewards.view(-1, self.num_gen_per_group).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_gen_per_group).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_gen_per_group, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_gen_per_group, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        
        per_token_loss = torch.exp(per_token_log_probs - per_token_log_probs.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        loss = per_token_loss.mean()
        return loss

    # todo: given a task, slip in the hindsight relabelled one.
    def _generate_completions(self, batch): 
        task = encode_task(batch[0], self.tokenizer, self. policy_model)
        input_ids = torch.tensor(task, device='cuda').unsqueeze(0)
        decoder_input_ids = self.policy_model.generate(input_ids, do_sample=True, num_return_sequences=self.num_gen_per_group) # do something here lets not worry for now
        rewards = self._compute_reward( batch[0], decoder_input_ids=decoder_input_ids) # will have to make this 
        return input_ids, decoder_input_ids, rewards 
    
        

    # pass in a list of hindsight relabelled tasks
    def train(self, tasks):
        
        # shuffle the order of the tasks.
        random.shuffle(tasks)
        
        for i in tqdm(range(0, len (tasks), self.batch_size)):
            self.optimizer.zero_grad()
            batch = tasks[i:i+self.batch_size]
            input_ids, decoder_input_ids, rewards = self._generate_completions(batch)
            loss = self._grpo_step(input_ids, decoder_input_ids, rewards)
            loss.backward()
            self.optimizer.step()

"""
Ok, so this is what needs to happen. I need to do the following.
Need to generate the completions. This part is fairly seperate.
Compute the reward for those completions.
so far so good.

then i compute the log probs of reference model
then i compute the log probs of the main model.

then there is some 

this is why they have the prompt length, because it is removing the prompt from the logit calcs.
"""



if __name__ == "__main__":
    task = Task.from_json("data/training/0a938d79.json")

    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    ref_model = T5ForConditionalGeneration.from_pretrained('finetune/2025-04-18_12-38-42/model')
    policy_model = T5ForConditionalGeneration.from_pretrained('finetune/2025-04-18_12-38-42/model')
    
    env = LineLevelArcEnv('Salesforce/codet5p-220m', 10, 1024, 1024, 5, 50000)
    ref_model.to('cuda')
    policy_model.to('cuda')

    trainer = GRPOTrainer(ref_model, 
                          policy_model, 
                          tokenizer,
                          env)
    
    task_array = [task]
    trainer.train(task_array)