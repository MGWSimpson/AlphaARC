
from collections import defaultdict
from dataclasses import dataclass
from torch.utils.data import DataLoader 
from torch.amp.grad_scaler import GradScaler
from torch import autocast
import torch
import torch.nn.functional as F
from tqdm import tqdm
from alphaarc.buffers import ReplayBuffer, TrajectoryBuffer
import torch.optim as optim

from alphaarc.logger import make_train_log, make_train_log_means, make_eval_log

class BaseTrainer:
    def __init__(self):
        pass

    def train(self, model, trajectory_buffer, supervised_buffer):
        raise NotImplementedError

    def train(self, model, replay_buffer): 
        raise NotImplementedError

class SampleFilterTrainer(BaseTrainer): 
    
    def __init__(self, lr, model, n_epochs=1, batch_size=8):
        super().__init__()


        self.scaler = GradScaler()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.n_epochs = n_epochs
        self.batch_size = 1

    def train(self, model, replay_buffer): 
        dataloader = DataLoader(replay_buffer, batch_size=self.batch_size, collate_fn=ReplayBuffer.collate_fn)

        for epoch in tqdm(range(self.n_epochs)):
            losses = []
            for input, target in tqdm( dataloader):
                self.optimizer.zero_grad()

                target[target == 0] = -1
                target[:, 0] = 0

                
                with autocast(device_type='cuda', dtype=torch.float16):
                    loss = model(input_ids=input.to(model.device),labels=target.to(model.device)).loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                losses.append(loss.detach().item())
            



class JointTrainer(BaseTrainer): 
    def __init__(self,   rl_batch_size, rl_lr, supervised_batch_size, supervised_lr,):
        super().__init__()

        self.learning_count = 0 

        self.rl_batch_size = rl_batch_size
        self.rl_lr = rl_lr
        self.supervised_batch_size = supervised_batch_size
        self.supervised_lr = supervised_lr
    
    def _train_rl(self, model, trajectory_buffer,batch_logs): 
        trajectory_dataloader = DataLoader(trajectory_buffer, 
                                           batch_size=self.rl_batch_size,
                                           collate_fn=TrajectoryBuffer.collate_fn)
        
        scaler = GradScaler()

            
                # Freeze LM head
        for param in model.model.lm_head.parameters():
            param.requires_grad = False

        # Create optimizer (backbone + heads)
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = optim.AdamW(trainable_params, lr=self.rl_lr) 

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
        replay_dataloader = DataLoader(supervised_buffer, batch_size=self.supervised_batch_size, collate_fn=ReplayBuffer.collate_fn)
        
            
        for param in model.model.lm_head.parameters():
            param.requires_grad = True


        optimizer = optim.AdamW(model.model.parameters(), self.supervised_lr) 

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

        torch.save(model, 'model.pth')

        train_log['supervised_buffer_capacity'] = len(supervised_buffer)
        train_log['rl_buffer_capacity'] = len(trajectory_buffer)
        return train_log


class MCTSTrainer(BaseTrainer): 
    def __init__(self):
        super().__init__()
        