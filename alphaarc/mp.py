import torch
import torch.multiprocessing as mp
import time
from torch.nn.utils.rnn import pad_sequence
from queue import Empty
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventType  # for type hints
import torch.nn.functional as F


class ModelRequester():
    def __init__(self, gpu_request_q, encode_request_q):
        self.gpu_request_q = gpu_request_q
        self.encode_request_q = encode_request_q
        self.read_conn , self.send_conn = mp.Pipe(duplex=False)

    def _make_gpu_request(self, task_data): 
        self.gpu_request_q.put((task_data, self.send_conn))
        result = self.read_conn.recv()

        tuple_to_return = ()
        for item in result:
            if type(item) is torch.Tensor:
                item = item.cpu().clone().numpy()
            tuple_to_return = tuple_to_return + (item, )
        
        return tuple_to_return

    def predict(self, task, state, past_key_values):
        return self._make_gpu_request((task.clone(), torch.tensor(state), past_key_values))

    
    def _make_encode_request(self, task, task_length): 
        task = torch.tensor(task).unsqueeze(0)
        self.encode_request_q.put((task, task_length, self.send_conn))
        result = self.read_conn.recv()

        return result.cpu()

    def encode(self, task, task_length): 
        return self._make_encode_request(task, task_length)

class ModelResponder(): 
    def __init__(self, gpu_request_q, encode_request_q, batch_size, model, load_model_event):
        
        self.gpu_request_q = gpu_request_q
        self.encode_request_q = encode_request_q
        self.batch_size = batch_size
        self.original_batch_size = batch_size
        self.model = model
        self.load_model_event = load_model_event
        self.time_out_time = 5

    def _handle_encode_request(self, request): 
        task, task_length, connection = request
        task_attention_mask = torch.ones((task.shape[0], task_length))
        task_attention_mask = F.pad(task_attention_mask, (0, task.shape[-1] - task_length), value=0)

        task = task.to(self.model.device)
        task_attention_mask = task_attention_mask.to(self.model.device)        

        with torch.no_grad(): 
            result = self.model.encode(task, task_attention_mask)
        result = result.last_hidden_state
        connection.send(result)

    def _handle_load_model(self): 
        self.model = torch.load('model.pth', weights_only=False)
        self.batch_size = self.original_batch_size

    def serve(self): 
        while True: 
            batch = []
            start_time = None
            while len(batch) < self.batch_size:
                
                if self.load_model_event.is_set():
                    self._handle_load_model()
                    self.load_model_event.clear()

                try:
                    encode_request = self.encode_request_q.get_nowait()
                    self._handle_encode_request(encode_request)
                except Empty: 
                    pass
                
                try: 
                    request = self.gpu_request_q.get_nowait()
                    batch.append(request)
                    if start_time  is None:
                        start_time = time.time()
                except Empty: 
                    pass
                
                if start_time is not None and (time.time() - start_time > self.time_out_time):
                    self.batch_size = len(batch)
                    break

            data, connections = zip(*batch)
            # packet everything up. and then pass it to the network class
            

            task, state, past_key_values = zip(*data)

            task = torch.stack(task).squeeze()
            if len(task.shape) == 1: 
                task = task.unsqueeze(0)




            state_attention_masks = [torch.ones(x.shape) for x in state]
            state_attention_masks = pad_sequence(state_attention_masks, batch_first=True)
            state = pad_sequence(state, batch_first=True)
            task_attention_masks =  (task != 0).bool()


            state_attention_masks = state_attention_masks.to(self.model.device)
            task_attention_masks = task_attention_masks.to(self.model.device) 
            task, state = task.to(self.model.device), state.to(self.model.device)
            results = self.model.predict(task,state, state_attention_masks, task_attention_masks, past_key_values)
            for i, connections in enumerate(connections):
                tuple_to_return = ()
                for j in range(len(results)):
                    tuple_to_return = tuple_to_return + (results[j][i], )

                connections.send(tuple_to_return)


@dataclass
class MultiProcessingContext:
    gpu_request_q:  mp.Queue
    encode_request_q:  mp.Queue
    trajectory_buffer_q:  mp.Queue
    replay_buffer_q:  mp.Queue
    episode_results_q:  mp.Queue
    task_q:  mp.JoinableQueue
    load_model_event: EventType




    




def build_mp_context(): 
    mp.set_start_method('spawn', force=True)
    gpu_request_q = mp.Queue ( )
    encode_request_q =  mp.Queue ( )
    trajectory_buffer_q =  mp.Queue ( )
    replay_buffer_q =  mp.Queue ( )
    episode_results_q =   mp.Queue ()
    task_q = mp.JoinableQueue()
    load_model_event = mp.Event()
    context = MultiProcessingContext(
                                     gpu_request_q=gpu_request_q,
                                     encode_request_q=encode_request_q, 
                                     trajectory_buffer_q=trajectory_buffer_q,
                                     replay_buffer_q=replay_buffer_q,
                                     episode_results_q=episode_results_q,
                                     task_q=task_q, 
                                     load_model_event=load_model_event)
    return context



def drain_q(q): 
    items = []
    while True:
        try:
            item = q.get_nowait()
            items.append(item)
        except Empty:
            break
    return items

def transfer_queues_to_buffers(trajectory_buffer, trajectory_q, replay_buffer, replay_q):
    trajectory_items = drain_q(trajectory_q)
    replay_items = drain_q(replay_q)

    for item in trajectory_items:
        trajectory_buffer.add_trajectory(item)

    for item in replay_items:
        task, program = item
        replay_buffer.add_program_and_task(task, program)
