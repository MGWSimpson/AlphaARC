
from collections import defaultdict
from dataclasses import dataclass
from torch.utils.data import DataLoader 
from torch.amp.grad_scaler import GradScaler
from torch import autocast
import torch
import torch.nn.functional as F
from tqdm import tqdm
from batchedalphaarc.buffers import ReplayBuffer, TrajectoryBuffer
from batchedalphaarc.logger import make_train_log, make_train_log_means, make_eval_log
import torch.optim as optim

class Trainer: 
    
    def __init__(self):
        self.learning_count = 0
 
    def _train_rl(self, model, trajectory_buffer,batch_logs): 
        trajectory_dataloader = DataLoader(trajectory_buffer, 
                                           batch_size=1,
                                           collate_fn=TrajectoryBuffer.collate_fn)
        
        scaler = GradScaler()

            
                # Freeze LM head
        for param in model.model.lm_head.parameters():
            param.requires_grad = False

        # Create optimizer (backbone + heads)
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = optim.AdamW(trainable_params) 

        for batch in tqdm(trajectory_dataloader, desc="rl training"):
            task, state, actions, target_pis, target_vs = batch
            optimizer.zero_grad()
            

            with autocast(device_type='cuda', dtype=torch.float16):
                target_pis = target_pis.to(model.device)
                target_vs = target_vs.to(model.device)
                predicted_pis, predicted_vs = model.forward( task.to(model.device), state.to(model.device), actions.to(model.device))

                policy_loss = F.cross_entropy(predicted_pis, target_pis)
                value_loss = F.mse_loss( predicted_vs, target_vs)
                
                loss = policy_loss + value_loss 
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            batch_logs["policy_batch_loss"].append(policy_loss.detach().item())
            batch_logs["value_batch_loss"].append(value_loss.detach().item())
            batch_logs['total_batch_loss'].append(loss.detach().item()) 


    def _train_supervised(self, model, supervised_buffer, batch_logs):
        scaler = GradScaler()
        replay_dataloader = DataLoader(supervised_buffer, batch_size=1, collate_fn=ReplayBuffer.collate_fn)
        
            
        for param in model.model.lm_head.parameters():
            param.requires_grad = True


        optimizer = optim.AdamW(model.model.parameters()) # TODO: need to add just the lower layers here.

        print(f"starting supervised training")
        for batch in tqdm(replay_dataloader, desc="supervised training"):
            optimizer.zero_grad()
            task, state = batch
            with autocast(device_type='cuda', dtype=torch.float16):
                loss = model.model(   input_ids=task.to(model.device),
                                    labels=state.to(model.device)).loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_logs["supervised_batch_loss"].append(loss.detach().item())


        




    def train(self, model, trajectory_buffer, supervised_buffer):

        train_log = make_train_log(self.learning_count)
        model.train()

        self._train_rl(model, trajectory_buffer, train_log)
        self._train_supervised(model, supervised_buffer, train_log)
        train_log =   make_train_log_means(train_log)
        self.learning_count +=1

        # make a save of the model.
        torch.save(model, 'model.pth')

        train_log['supervised_buffer_capacity'] = len(supervised_buffer)
        train_log['rl_buffer_capacity'] = len(trajectory_buffer)
        return train_log

    