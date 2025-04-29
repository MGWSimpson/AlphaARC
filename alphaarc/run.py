import argparse

from alphaarc.configs import load_config, AlphaARCConfig
from alphaarc.mp import build_mp_context, MultiProcessingContext, transfer_queues_to_buffers, drain_q
from alphaarc.curriculum import BaseCurriculum
from alphaarc.train import BaseTrainer
from alphaarc.buffers import TrajectoryBuffer, ReplayBuffer
from tqdm import tqdm
import wandb
from dataclasses import dataclass, asdict



def run_experiment( config: AlphaARCConfig, 
                    curriculum: BaseCurriculum, 
                    trainer: BaseTrainer,
                    trajectory_buffer: TrajectoryBuffer, 
                    replay_buffer: ReplayBuffer,
                    mp_context: MultiProcessingContext, 
                    model:  
                    run): 
    

    for meta_epoch in config.n_epochs:
        full_curriculum = curriculum.generate_curriculum()
        
        for i in tqdm(range(0, len(full_curriculum), config.train_every)):

            curriculum_chunk = full_curriculum[i:i + config.train_every]
            for task in curriculum_chunk:
                mp_context.curriculum_q.put(task, block=True)
            
            mp_context.curriculum_q.join()
            transfer_queues_to_buffers(trajectory_buffer=trajectory_buffer,
                                       trajectory_q=mp_context.trajectory_buffer_q,
                                       replay_buffer=replay_buffer,
                                       replay_q=mp_context.replay_buffer_q
                                       )

            # TODO: make sure you add this back in current_batch_size.value = n_tree_workers 
            episode_logs = drain_q(mp_context. episode_results_q)
            train_log = trainer.train(model=model, trajectory_buffer=trajectory_buffer, supervised_buffer=replay_buffer)
            mp_context.load_model_event.set()
            

            
            print("starting evaluation.")
            eval_log, eval_episode_logs = evaluate(eval_set, curriculum_q, episode_results_q)

            run_log = make_run_log(train_log, episode_logs, eval_log, eval_episode_logs)
            run.log(run_log)


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')

    args = parser.parse_args()
    config = load_config()
    
    mp_context = build_mp_context()
    
    run = wandb.init(
    project="alphaarc",
    config=asdict(config))


if __name__ == "__main__": 
    main()