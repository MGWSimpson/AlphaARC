import torch
import torch.multiprocessing as mp
import time
from torch.nn.utils.rnn import pad_sequence
from queue import Empty

from dataclasses import dataclass

class ModelRequester():

    def __init__(self, gpu_request_q, encode_request_q):
        self.gpu_request_q = gpu_request_q
        self.encode_request_q = encode_request_q
        self.read_conn , self.send_conn = mp.Pipe(duplex=False)

    def _make_gpu_request(self, task_data): 
        self.gpu_request_q.put_nowait((task_data, self.send_conn))
        result = self.read_conn.recv()
        actions, action_probs, value, child_key_values = result
        actions, action_probs, value, child_key_values = actions.cpu(), action_probs.cpu(), value.cpu(), child_key_values
        return actions.numpy(), action_probs.numpy(), value.numpy(), child_key_values

    def predict(self, task, state, past_key_values):
        return self._make_gpu_request((task.clone(), torch.tensor(state), past_key_values))

    
    def _make_encode_request(self, task): 
        task = torch.tensor(task).unsqueeze(0)
        self.encode_request_q.put_nowait((task, self.send_conn))
        result = self.read_conn.recv()

        return result.cpu()

    def encode(self, task): 
        return self._make_encode_request(task)

class ModelResponder(): 
    def __init__(self, gpu_request_q, encode_request_q, batch_size, model, load_model_event):
        
        self.gpu_request_q = gpu_request_q
        self.encode_request_q = encode_request_q
        self.batch_size = batch_size
        self.model = model

        self.load_model_event = load_model_event

        self.time_out_time = 1

    def _handle_encode_request(self, request): 
        task, connection = request
        task = task.to(self.model.device)
        with torch.no_grad(): 
            result = self.model.model.encoder(task)
        result = result.last_hidden_state
        connection.send(result)

    def _handle_load_model(self): 
        self.model = torch.load('model.pth', weights_only=False)

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
                        # If timeout has started and expired, break
                
                if start_time is not None and (time.time() - start_time > self.time_out_time):
                    self.batch_size = len(batch)
                    break
                    
            data, connections = zip(*batch)
            # packet everything up. and then pass it to the network class
            
            task, state, past_key_values = zip(*data)
            task = pad_sequence(task, batch_first=True)
            state = pad_sequence(state, batch_first=True)


            task, state = task.to(self.model.device), state.to(self.model.device)
            actions, action_probs ,values, past_key_values = self.model.predict(task, state, past_key_values)


            if len(batch) == 1:
                action_probs = action_probs.unsqueeze(0)

            for i, connections in enumerate(connections):
                connections.send(  ( actions[i], 
                                    action_probs[i], 
                                    values[i],
                                    past_key_values))




@dataclass
class MultiProcessingContext:
    curriculum_q = mp.JoinableQueue()
    gpu_request_q = mp.Queue()
    encode_request_q = mp.Queue()
    trajectory_buffer_q = mp.Queue()
    replay_buffer_q = mp.Queue()
    episode_results_q = mp.Queue()
    task_q = mp.Queue()
    load_model_event = mp.Event()




def build_mp_context(): 
    context = MultiProcessingContext()
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
