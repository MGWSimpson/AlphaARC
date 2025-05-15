"""
minimal implementation of GRPO. Probably not super efficient but hacked together for my need.
reference implementation: https://github.com/lsdefine/simple_GRPO/
"""
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
import random
from tqdm import tqdm
import torch

class GRPOTrainer:

    def __init__(self,
                 ref_model: T5ForConditionalGeneration,
                 policy_model: T5ForConditionalGeneration,
                 tokenizer, 
                 num_gen_per_group=8,
                 batch_size=2, 
                 lr=1e-6,
                 beta= 0.04): 
        
        self.tokenizer = tokenizer
        self.optimizer = AdamW(policy_model.parameters(), lr=lr)
        
        self.ref_model = ref_model
        self.ref_model.requires_grad_(False)
        
        self.policy_model = policy_model
        self.num_gen_per_group = num_gen_per_group # we will sneak in the correct answer !



    def _compute_logits(self, model): 
        pass

    
    def _compute_per_token_log_probs(self,): 
        pass

    # returns the loss
    def _grpo_step(self, input_ids, decoder_input_ids, rewards): 
        

        # get the logits :D 
        policy_logits =  self._compute_logits(self.policy_model )
        ref_logits = self._compute_logits(self.ref_model)


        # compute the log probs
        per_token_log_probs = self._compute_per_token_log_probs()
        ref_per_token_log_probs = self._compute_per_token_log_probs()


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

    def _generate_completions(self, batch):
        input_ids = None # do something here to the batch, oh and slip in the right answers
        decoder_input_ids = self.policy_model.generate(input_ids) # do something here lets not worry for now
        rewards = None
        return input_ids, decoder_input_ids, rewards 
    
        

    # pass in a list of hindsight relabelled tasks
    def train(self, tasks):
        tasks = random.shuffle(tasks)
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