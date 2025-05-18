"""
minimal implementation of GRPO. Probably not super efficient but hacked together for my need.
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
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from alphaarc.utils import relabel_task
import wandb
import pytorch_lightning as pl

from pprint import pprint

def encode_task(task, tokenizer, model, input_state_max=256, n_examples=10, max_length=256): 
    tokenized_task = np.array(tokenize_task(task, tokenizer, n_examples, input_state_max, max_length)['input_ids'])
    return tokenized_task




def compute_logits(input_ids, decoder_input_ids, model): 
    # use loop to reduce memory
    logits = []
    for b in range(decoder_input_ids.shape[0]):
        logits.append(model(input_ids=input_ids, 
                            decoder_input_ids=decoder_input_ids[b, :].unsqueeze(0)).logits)

    logits =torch.stack(logits, dim=0).squeeze()
    return logits


def compute_per_token_log_probs(logits, decoder_input_ids): 
    per_token_log_probs = []
    for logits_row, input_ids_row in zip(logits, decoder_input_ids): 
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_log_probs.append(token_log_prob)

    return torch.stack(per_token_log_probs)



def encode_program(program_lines, tokenizer): 
    tokenized_program = tokenizer(program_lines, return_tensors='pt')['input_ids']
    tokenized_program = torch.concat((torch.tensor([0]).unsqueeze(0), tokenized_program), dim= -1)
    return tokenized_program



# need to pad them together basically
def combine_her(decoder_input_ids, her_tensor):
    max_len = max(decoder_input_ids.shape[-1], her_tensor.shape[-1])

    decoder_input_ids = F.pad( decoder_input_ids, (0, max_len- decoder_input_ids.shape[-1]), 'constant', 0 )
    her_tensor = F.pad(  her_tensor, (0, max_len- her_tensor.shape[-1]), 'constant', 0 )

    return torch.concat((decoder_input_ids, her_tensor.to('cuda')), dim=0)

def append_her_answer(decoder_input_ids, task: Task, tokenizer): 
    program_lines = task.program_lines
    encoded_program = encode_program(program_lines, tokenizer)
    return combine_her(decoder_input_ids, encoded_program)





class GRPOTrainer:

    def __init__(self,
                 ref_model: T5ForConditionalGeneration,
                 policy_model: T5ForConditionalGeneration,
                 tokenizer: AutoTokenizer,
                 env: BaseEnv,
                 num_gen_per_group=8,
                 batch_size=2, 
                 lr=5e-6,
                 beta= 0.04,
                 clip_param = 0.2): 
        
        self.tokenizer = tokenizer
        self.optimizer = AdamW(policy_model.parameters(), lr=lr)
        

        self.grad_accumulate_steps = 8

        self.grad_accumulate_cntr = 0

        self.ref_model = ref_model
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()

        self.policy_model = policy_model
        self.env = env
        

        # grpo params
        self.num_gen_per_group = num_gen_per_group # we will sneak in the correct answer ! -> hindsight stuff
        self.batch_size = batch_size
        self.beta = beta
        self.clip_param = clip_param


        self.run = wandb.init(
            project="alphaarc")


    def _relabel(self, input_ids, decoder_input_ids, all_ids, task, env):
        new_task = relabel_task(task[0], env, decoder_input_ids, self.tokenizer.decode(decoder_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        input_ids = encode_task(new_task, self.tokenizer, self.ref_model)


        new_rewards = self._compute_reward(new_task, all_ids)
        return torch.tensor(input_ids, device='cuda').unsqueeze(0), new_rewards, new_task
        

    """
    Need to change this over to be multiple decoder input ids
    """
    def _compute_reward(self, task, decoder_input_ids): 
        self.env.set_task(task)
        rewards = [self.env.evaluate_program(x, should_token_account=False)[0] for x in decoder_input_ids]
        return torch.tensor(rewards, dtype=torch.float, device='cuda')
        

    def _compute_exploration_reward(self, task, decoder_input_ids): 
        self.env.set_task(task)
        rewards = [self.env.evaluate_program(x, should_token_account=False)[0] for x in decoder_input_ids]
        correctness_reward = [0.1 if self.env.is_valid_syntax(x) else -1 for x in decoder_input_ids]

        rewards = [max(a, b) for a, b in zip(rewards, correctness_reward)]

        return torch.tensor(rewards, dtype=torch.float, device='cuda')
    


    # returns the loss
    def _grpo_step(self, input_ids, decoder_input_ids, rewards): 
        

        # get the logits :D 
        policy_logits =  compute_logits(input_ids, decoder_input_ids, self.policy_model )
        ref_logits = compute_logits(input_ids, decoder_input_ids, self.ref_model)




        # compute the log probs
        per_token_log_probs = compute_per_token_log_probs(policy_logits, decoder_input_ids)
        ref_per_token_log_probs = compute_per_token_log_probs(ref_logits, decoder_input_ids)




        # kl
        per_token_kl = torch.exp(ref_per_token_log_probs - per_token_log_probs) - (ref_per_token_log_probs - per_token_log_probs) - 1


        # normalize the generation 
        mean_grouped_rewards = rewards.view(-1, self.num_gen_per_group).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_gen_per_group).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_gen_per_group, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_gen_per_group, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages = advantages.unsqueeze(1)


        """if advantages.all() == 1: # replace advantage with negative advantage -> should be changed to if enabled.
            advantages[advantages == 1] = 1

        if advantages.all() == 0: # replace advantage with negative advantage -> should be changed to if enabled.
            advantages[advantages == 0] = -1"""

        completion_mask = (decoder_input_ids != 0).int()
        

        
        

        ratio = torch.exp(per_token_log_probs - ref_per_token_log_probs.to('cuda'))
        clipped_ratio = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)


        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
        

        
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # for logging
        clipped_mask = (ratio > 1 + self.clip_param) | (ratio < 1 - self.clip_param)    # boolean mask
        
        return loss, per_token_kl, advantages, clipped_mask, rewards

    # todo: given a task, slip in the hindsight relabelled one.
    def _generate_completions(self, batch, should_append_HER=False, exploration_reward=False): 
        task = encode_task(batch[0], self.tokenizer, self. policy_model)
        input_ids = torch.tensor(task, device='cuda').unsqueeze(0)
        
        if should_append_HER:
            num_gen_per_group  = self.num_gen_per_group - 1
        else:
            num_gen_per_group = self.num_gen_per_group


        with torch.inference_mode():
            decoder_input_ids = self.policy_model.generate(input_ids, max_new_tokens=512, do_sample=True, 
                                                            num_return_sequences=num_gen_per_group) # generate it with -1 to account for the HER example
        
        decoder_input_ids = decoder_input_ids.clone()
        if should_append_HER:
            decoder_input_ids = append_her_answer(decoder_input_ids, batch[0], self. tokenizer)
        

        if exploration_reward:
            rewards = self._compute_exploration_reward( batch[0], decoder_input_ids=decoder_input_ids) # will have to make this 
        else:
            rewards = self._compute_reward( batch[0], decoder_input_ids=decoder_input_ids) # will have to make this 
        return input_ids, decoder_input_ids, rewards 
    
        

    # pass in a list of hindsight relabelled tasks
    def train(self, tasks):
        self.policy_model.eval()
        self.ref_model.eval()

        # shuffle the order of the tasks.
        random.shuffle(tasks)
        for i in tqdm(range(0, len (tasks), self.batch_size)):
            self.optimizer.zero_grad()
            batch = tasks[i:i+self.batch_size]
            input_ids, decoder_input_ids, rewards = self._generate_completions(batch)
            loss = self._grpo_step(input_ids, decoder_input_ids, rewards)
            loss.backward()
            self.optimizer.step()


    def generate_answers(self, task, n_generations):
        answers = []
        for i in range(0, n_generations, self.num_gen_per_group):
            batch = task
            input_ids, decoder_input_ids, rewards = self._generate_completions(batch, exploration_reward=False)
            loss, ptkl, adv, clp_mask, rwrd = self._grpo_step(input_ids, decoder_input_ids, rewards)


          

            grad_norm = torch.nn.utils.get_total_norm(self.policy_model.parameters())
                # log metrics
            self.run.log({  "loss": loss.detach().cpu().item(),
                                "percent_clipped": clp_mask.cpu().float().mean().item(),
                                "advantage": adv.cpu().mean().item(),
                                "ptkl": ptkl.cpu().mean().item(),
                                "grad norm": grad_norm.item(),
                                "avg reward": rwrd.cpu().mean().item()})
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            """x = self.tokenizer( task[0].program_lines, add_special_tokens=False, return_tensors='pt')['input_ids']
            x = x.to('cuda')
            
            with torch.inference_mode():
                policy_logits =  compute_logits(input_ids, x, self.policy_model ).unsqueeze(0)
                answer_log_probs = compute_per_token_log_probs(policy_logits, x)


            print("-----")
            # print(self.tokenizer.batch_decode(decoder_input_ids))
            print(task[0].program_lines)
            print(f"sum log prob of answer: {answer_log_probs.cpu().sum().item()}")
            print("-----")"""
            answers.extend([x for x in decoder_input_ids])  

        answer_tensor = pad_sequence(answers, batch_first=True)
        return answer_tensor



    def generate_relabeled_answers_internal(self, task, env, n_generations): 
        answers = []
        self.env.set_task(task[0])
        for i in range(0, n_generations, self.num_gen_per_group):
            batch = task
            input_ids, decoder_input_ids, rewards = self._generate_completions(batch)

            loss, ptkl, adv, clp_mask, rwrd = self._grpo_step(input_ids, decoder_input_ids, rewards)


            grad_norm = torch.nn.utils.get_total_norm(self.policy_model.parameters())
                # log metrics
            self.run.log({  "loss": loss.detach().cpu().item(),
                                "percent_clipped": clp_mask.cpu().float().mean().item(),
                                "advantage": adv.cpu().mean().item(),
                                "ptkl": ptkl.cpu().mean().item(),
                                "grad norm": grad_norm.item(),
                                "avg reward": rwrd.cpu().mean().item()})
            
            loss.backward()
            


            # you would basically insert it here, for each of the decoder ids, you would relabel it
            for i in range(decoder_input_ids.shape[0]):
                

                if not env.is_valid_syntax(decoder_input_ids[i]): # if not valid program, skip relabel
                    continue 
                
                input_ids, rewards, new_task = self._relabel(input_ids, decoder_input_ids[i], decoder_input_ids, task, env)    
                
                loss, ptkl, adv, clp_mask, rwrd = self._grpo_step(input_ids, decoder_input_ids, rewards)
                loss.backward()
                
                # self.grad_accumulate_cntr += 1

                # if self.grad_accumulate_cntr % self.grad_accumulate_steps  == 0:
                    # self.grad_accumulate_cntr = 0

                grad_norm = torch.nn.utils.get_total_norm(self.policy_model.parameters())
                # log metrics
                self.run.log({  "loss": loss.detach().cpu().item(),
                                "percent_clipped": clp_mask.cpu().float().mean().item(),
                                "advantage": adv.cpu().mean().item(),
                                "ptkl": ptkl.cpu().mean().item(),
                                "grad norm": grad_norm.item(),
                                "avg reward": rwrd.cpu().mean().item()})

            self.optimizer.step()  
            self.optimizer.zero_grad()

            x = self.tokenizer( task[0].program_lines, add_special_tokens=False, return_tensors='pt')['input_ids']
            x = x.to('cuda')
            
            with torch.inference_mode():
                policy_logits =  compute_logits(input_ids, x, self.policy_model ).unsqueeze(0)
                answer_log_probs = compute_per_token_log_probs(policy_logits, x)

            print("-----")
            # print(self.tokenizer.batch_decode(decoder_input_ids))
            print(task[0].program_lines)
            print(f"sum log prob of answer: {answer_log_probs.cpu().sum().item()}")
            print("-----")


            answers.extend([x for x in decoder_input_ids])  

        answer_tensor = pad_sequence(answers, batch_first=True)
        self.env.set_task(task[0])
        return answer_tensor
    


    def generate_relabeled_answers_external(self, task, env, n_generations): 
        answers = []
        self.env.set_task(task[0])
        for i in range(0, n_generations, self.num_gen_per_group):
            batch = task
            input_ids, decoder_input_ids, rewards = self._generate_completions(batch)

            loss, ptkl, adv, clp_mask, rwrd = self._grpo_step(input_ids, decoder_input_ids, rewards)


            grad_norm = torch.nn.utils.get_total_norm(self.policy_model.parameters())
                # log metrics
            self.run.log({  "loss": loss.detach().cpu().item(),
                                "percent_clipped": clp_mask.cpu().float().mean().item(),
                                "advantage": adv.cpu().mean().item(),
                                "ptkl": ptkl.cpu().mean().item(),
                                "grad norm": grad_norm.item(),
                                "avg reward": rwrd.cpu().mean().item()})
            
            loss.backward()


            # you would basically insert it here, for each of the decoder ids, you would relabel it
            for i in range(decoder_input_ids.shape[0]):
                

                if not env.is_valid_syntax(decoder_input_ids[i]): # if not valid program, skip relabel
                    continue 
                
                input_ids, rewards, new_task = self._relabel(input_ids, decoder_input_ids[i], decoder_input_ids, task, env)   
                new_task = [new_task] 
                input_ids, new_decoder_input_ids, rewards= self._generate_completions(new_task, should_append_HER=True)

                loss, ptkl, adv, clp_mask, rwrd = self._grpo_step(input_ids, new_decoder_input_ids, rewards)
                loss.backward()
                
                # self.grad_accumulate_cntr += 1

                # if self.grad_accumulate_cntr % self.grad_accumulate_steps  == 0:
                    # self.grad_accumulate_cntr = 0

                grad_norm = torch.nn.utils.get_total_norm(self.policy_model.parameters())
                # log metrics
                self.run.log({  "loss": loss.detach().cpu().item(),
                                "percent_clipped": clp_mask.cpu().float().mean().item(),
                                "advantage": adv.cpu().mean().item(),
                                "ptkl": ptkl.cpu().mean().item(),
                                "grad norm": grad_norm.item(),
                                "avg reward": rwrd.cpu().mean().item()})

            self.optimizer.step()  
            self.optimizer.zero_grad()
                    
            answers.extend([x for x in decoder_input_ids])  

        answer_tensor = pad_sequence(answers, batch_first=True)
        self.env.set_task(task[0])
        return answer_tensor




if __name__ == "__main__":
    task = Task.from_json("data/training/b527c5c6.json")
    pl.seed_everything(0)
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    ref_model = T5ForConditionalGeneration.from_pretrained('finetune/2025-04-18_12-38-42/model')
    policy_model = T5ForConditionalGeneration.from_pretrained('finetune/2025-04-18_12-38-42/model')
    
    env = LineLevelArcEnv('Salesforce/codet5p-220m', 10, 256, 256, 5, 50000)
    ref_model.to('cuda')
    policy_model.to('cuda')

    trainer = GRPOTrainer(ref_model, 
                          policy_model, 
                          tokenizer,
                          env)
    
    task_array = [task]
    trainer.train(task_array)