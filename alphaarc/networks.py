import torch
import torch.nn as nn

from transformers import T5ForConditionalGeneration
    

class PolicyValueNetwork(nn.Module): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5p-220m')
        self.base_model = model.base_model
        self.policy = model.lm_head
        self.value = nn.Linear(768, 1)    



    # training mode (compute the log probs)
    def forward(self, state):
        pass

    # inference mode
    def predict(self, state): 
        pass

    def train(self, replay_buffer): 
        pass

if __name__ == "__main__":
    test = PolicyValueNetwork()

    print(test.policy)