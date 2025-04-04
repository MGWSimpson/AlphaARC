import torch
import torch.nn as nn

from transformers import T5ForConditionalGeneration

"""
Joint policy value network.
"""
class PolicyValueNetwork(nn.Module):
    def __init__(self):
        self.policy = T5ForConditionalGeneration.from_pretrained('codet5p-220m')

        self.num_samples = 5
        self.temperature = 0.95
        self.max_length = 512
        self.stop_strings = "\n"


    def predict(self, state):
        pass

    def forward(self, input_ids, attention_mask): 
        outputs = self.network.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        do_sample=True,
                                        temperature=self.temperature,
                                        max_length=self.max_length,
                                        num_return_sequences=self.num_samples,
                                        output_scores=True,
                                        stop_strings=self.stop_strings) # we just want a single line
        values = torch.ones((outputs.shape))
        return outputs, values