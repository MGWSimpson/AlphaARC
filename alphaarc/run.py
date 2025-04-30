import argparse
from multiprocessing import Process  
from alphaarc.configs import load_config, AlphaARCConfig
from alphaarc.mp import build_mp_context, MultiProcessingContext, transfer_queues_to_buffers, drain_q
from alphaarc.curriculum import BaseCurriculum
from alphaarc.train import BaseTrainer
from alphaarc.buffers import TrajectoryBuffer, ReplayBuffer
from alphaarc.networks import BaseNetwork
from alphaarc.logger import make_run_log, make_train_only_run_log,  summarize_episodes, make_eval_log
from tqdm import tqdm
import wandb
from dataclasses import dataclass, asdict
import traceback
from alphaarc.mp import ModelRequester, ModelResponder
from alphaarc.agent import Agent
from alphaarc.env import BaseEnv
from alphaarc.configs import build_alpha_arc_config, build_network, build_env, build_policy, build_curriculum, build_trainer

def tree_worker_fn(config, 
                   mp_context: MultiProcessingContext,
                   ): 
    

    model_requester = ModelRequester(gpu_request_q=mp_context.gpu_request_q,
                                     encode_request_q=mp_context.encode_request_q)
    env = build_env(config['env_config'])
    policy = build_policy(model_requester, env, config['policy_config'])
    agent = Agent(policy=policy, replay_q=mp_context.replay_buffer_q, trajectory_q=mp_context.trajectory_buffer_q)
    
    try:
        while True:
            task = mp_context.task_q.get()
            env.set_task(task)
            
            if task.is_eval:
                result = agent.evaluate(env)
            else:
                result = agent.learn(env) 
            
            mp_context.episode_results_q.put_nowait(result)
            mp_context.task_q.task_done()
    
    except Exception as e:
        print(f"[CPU THREAD ERROR] {e}")
        traceback.print_exc()




def gpu_worker_fn(model_responder: ModelResponder): 
    try:
        model_responder.serve()
    except Exception as e:
        print(f"[GPU THREAD ERROR] {e}")
        traceback.print_exc()
        

def evaluate(eval_set, task_q, episode_results_q): 
    eval_log = make_eval_log()
    
    # enqueue the curriculum.
    for task in eval_set.generate_curriculum():
        task_q.put(task, block=True)
    task_q.join()
    episode_logs = drain_q(episode_results_q)
    
    eval_log['solve_rate'] = summarize_episodes(episode_logs)
    return eval_log, episode_logs


def run_experiment( config: AlphaARCConfig, 
                    curriculum: BaseCurriculum, 
                    evaluation_set: BaseCurriculum,
                    trainer: BaseTrainer,
                    trajectory_buffer: TrajectoryBuffer, 
                    replay_buffer: ReplayBuffer,
                    mp_context: MultiProcessingContext, 
                    model: BaseNetwork, 
                    run): 
    

    for meta_epoch in range(config.n_epochs):
        full_curriculum = curriculum.generate_curriculum()

            
        for i in tqdm(range(0, len(full_curriculum), config.train_every)):
            curriculum_chunk = full_curriculum[i:i + config.train_every]

            for task in curriculum_chunk:
                mp_context.task_q .put(task, block=True)
            
            mp_context.task_q.join()
            transfer_queues_to_buffers(trajectory_buffer=trajectory_buffer,
                                       trajectory_q=mp_context.trajectory_buffer_q,
                                       replay_buffer=replay_buffer,
                                       replay_q=mp_context.replay_buffer_q
                                       )

            episode_logs = drain_q(mp_context. episode_results_q)
            train_log = trainer.train(model=model, trajectory_buffer=trajectory_buffer, supervised_buffer=replay_buffer)
            mp_context.load_model_event.set()
            run_log = make_train_only_run_log(train_log, episode_logs)
            run.log(run_log)


def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='alphaarc/configs/base_config.yaml')
    args = parser.parse_args()

    config = load_config(args.config_path)
    
    alpha_arc_config = build_alpha_arc_config(config['alpha_arc_config'])
    mp_context = build_mp_context()
    

    model = build_network(config['model_config'])
    model = model.to(model.device)

    model_responder = ModelResponder(gpu_request_q=mp_context.gpu_request_q,
                                     encode_request_q=mp_context.encode_request_q,
                                     batch_size=alpha_arc_config.n_tree_workers,
                                     load_model_event=mp_context.load_model_event,
                                     model=model)
    

    run = wandb.init(
    project="alphaarc",
    config=config)

    gpu_worker = Process(target=gpu_worker_fn, args=(model_responder, ))
    tree_workers = [Process(target=tree_worker_fn, args=(config, mp_context,), daemon=True) for _ in range(alpha_arc_config.n_tree_workers)]

    gpu_worker.start()

    # spin up workers
    for worker in tree_workers:
        worker.start()


    
    trainer = build_trainer(config['trainer_config'])

    curriculum = build_curriculum(config['training_curriculum_config'])
    eval_set = build_curriculum(config['evaluation_curriculum_config'] )

    trajectory_buffer = TrajectoryBuffer()
    replay_buffer = ReplayBuffer()
    # run experiment
    run_experiment(alpha_arc_config,
                   curriculum, 
                   eval_set, 
                   trainer,
                   trajectory_buffer,
                   replay_buffer,
                   mp_context,
                   model,
                   run
                   )

    [worker.kill() for worker in tree_workers]
    gpu_worker.kill()
    # run.finish()
    print("workers done")
    print("all done!")

    
    
    

if __name__ == "__main__": 
    main()