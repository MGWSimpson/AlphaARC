import torch
import multiprocessing as mp
from multiprocessing import Process
from batchedalphaarc.agent import Agent
from batchedalphaarc.curriculum import Curriculum
from queue import Empty
from batchedalphaarc.env import LineLevelArcEnv
from dataclasses import dataclass
from transformers import T5ForConditionalGeneration, AutoTokenizer



from batchedalphaarc.env import LineLevelArcEnv
from batchedalphaarc.curriculum import Curriculum
from batchedalphaarc.agent import Agent
from batchedalphaarc.buffers import ReplayBuffer, TrajectoryBuffer
from batchedalphaarc.networks import PolicyValueNetwork
from batchedalphaarc.logger import Logger


@dataclass
class RLTrainingConfig:
    rl_batch_size: int =2

@dataclass
class SupervisedTrainingConfig:
    supervised_batch_size: int = 2

@dataclass
class ModelConfig:
    model_path: str = 'finetune/2025-04-18_12-38-42/model'
    tokenizer_path: str = 'Salesforce/codet5p-220m'
    model_temperature: float = 0.95
    device: str = 'cuda'


@dataclass
class batchedalphaarcConfig:
    rl_training_config: RLTrainingConfig = RLTrainingConfig()
    supervised_training_config: SupervisedTrainingConfig = SupervisedTrainingConfig()
    model_config: ModelConfig = ModelConfig()
    n_actions: int = 5
    n_examples: int = 10
    n_episodes_per_task: int = 1
    n_simulations: int = 10
    action_temperature: float = 1
    seed: int = 0
    max_state_len: int = 1024
    max_task_len: int = 512
    max_action_len: int = 20
    trajectory_buffer_capacity = 100_000
    replay_buffer_capacity: int = 100_000
    train_every: int = 100


def make_gpu_request(gpu_request_q, task_data): 
    parent_conn, child_conn =mp.Pipe(duplex=False)
    gpu_request_q.put_nowait((task_data, child_conn))
    print("awaiting response....")
    result = parent_conn.recv()
    print("received response....")


def tree_worker_fn(task_q: mp.JoinableQueue, gpu_request_q: mp.Queue): 
    while True:
        task = task_q.get()
        task_data = torch.randint(0, 255, (1, 128))
        result = make_gpu_request(gpu_request_q, task_data)
        task_q.task_done()
         




def gpu_worker_fn(gpu_request_q: mp.Queue, batch_size=4): 
    while True: 
        batch = []

        while len(batch) < batch_size:
            request = gpu_request_q.get()
            batch.append(request)
        
        print("processing batch!")

        # do some computation

        for item in batch:
            task_data, conn = item
            conn.send(task_data)

 
    


"""
Model -> model queue.
Buffers -> locks and stuff.

"""
if __name__ == "__main__": 
    n_tree_workers = 4
    
    
    curriculum = Curriculum(dir_paths=['data/evaluation'])
    curriculum_q = mp.JoinableQueue(maxsize=len(curriculum))
    gpu_request_q = mp.Queue()

    gpu_worker = Process(target=gpu_worker_fn, args=(gpu_request_q, ))
    gpu_worker.start()

    tree_workers = [Process(target=tree_worker_fn, args=(curriculum_q, gpu_request_q), daemon=True) for _ in range(n_tree_workers)]

    # can break up the eval by just passing in these queues basically.
    for task in curriculum.generate_curriculum():
        curriculum_q.put(task, block=True)

    

    for worker in tree_workers:
        worker.start()
    
    curriculum_q.join()
    gpu_worker.kill()
    print("workers done")
    print("all done!")