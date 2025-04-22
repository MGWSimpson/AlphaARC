import multiprocessing as mp
from multiprocessing import Process
from batchedalphaarc.agent import Agent
from batchedalphaarc.curriculum import Curriculum
from queue import Empty

def tree_worker_fn(task_queue: mp.JoinableQueue): 
    while True:
        task = task_queue.get()
        print(task)
        task_queue.task_done()
         




def gpu_worker_fn(): 
    # keep checking if there exists some
    # 
    #
    # 
    pass

 
    
   
    


if __name__ == "__main__": 
    n_tree_workers = 4
    curriculum = Curriculum(dir_paths=['data/evaluation'])
    curriculum_q = mp.JoinableQueue(maxsize=len(curriculum))
    tree_workers = [Process(target=tree_worker_fn, args=(curriculum_q, ), daemon=True) for _ in range(n_tree_workers)]

    for task in curriculum.generate_curriculum():
        curriculum_q.put(task, block=True)

    

    for worker in tree_workers:
        worker.start()
    
    curriculum_q.join()
    print("workers done")
    print("all done!")