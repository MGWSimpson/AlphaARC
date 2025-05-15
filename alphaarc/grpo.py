"""
minimal implementation of GRPO. Probably not super efficient but hacked together for my need.
"""
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
import random
from tqdm import tqdm

class GRPOTrainer:

    def __init__(self,
                 ref_model: T5ForConditionalGeneration,
                 policy_model: T5ForConditionalGeneration,
                 tokenizer, 
                 num_gen_per_group=8,
                 batch_size=2, 
                 lr=1e-6,): 
        
        self.tokenizer = tokenizer
        self.optimizer = AdamW(policy_model.parameters(), lr=lr)
        self.ref_model = ref_model
        ref_model.requires_grad_(False)
        self.policy_model = policy_model
        self.num_gen_per_group = num_gen_per_group # we will sneak in the correct answer !



    # returns the loss
    def _grpo_step(self, input_ids, decoder_input_ids): 
        pass


    def _generate_completions(self, batch):
        input_ids = None # do something here to the batch, oh and slip in the right answers
        decoder_input_ids = self.policy_model.generate(input_ids) # do something here lets not worry for now
        return input_ids, decoder_input_ids 
    
        

    # pass in a list of hindsight relabelled tasks
    def train(self, tasks):
        tasks = random.shuffle(tasks)
        for i in tqdm(range(0, len (tasks), self.batch_size)):
            self.optimizer.zero_grad()
            batch = tasks[i:i+self.batch_size]
            input_ids, decoder_input_ids = self._generate_completions(batch)
            loss = self._grpo_step(input_ids, decoder_input_ids)
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