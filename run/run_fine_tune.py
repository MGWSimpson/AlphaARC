# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import cProfile
import csv
import gc
import json
import os
import time
from typing import Any

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from codeit.agent import Agent, calculate_performance
from codeit.callbacks import HfModelCheckpoint
from codeit.exit_data_module import ExItDataModule
from codeit.hf_model_module import HFModule
from codeit.task import from_dict
from codeit.utils import get_num_pixels

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def calculate_performance_over_inference_tasks(solutions, inference_keys):
    task_demonstration_performance_list = []
    test_performance_list = []
    for task_key in inference_keys:
        task_demonstration_performance, test_performance = calculate_performance(
            solutions, task_key
        )
        task_demonstration_performance_list.append(task_demonstration_performance)
        test_performance_list.append(test_performance)
    return {
        "task_demonstration_performance": task_demonstration_performance_list,
        "test_performance": test_performance_list,
    }


def filter_by_inference_keys(tasks, inference_keys):
    task_keys = [
        s for s in tasks.keys() if not any(s.startswith(prefix) for prefix in inference_keys)
    ]
    filtered_tasks = {}
    for key in task_keys:
        filtered_tasks[key] = tasks[key]
    return filtered_tasks


def initialise_csv_writer(config):
    results_dir = config.run_dir + "/performance.csv"
    with open(results_dir, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "meta_iteration",
                "cumulative_performance",
                "performance",
                "step",
                "num_mutated_tasks",
                "num_policy_tasks",
            ]
        )


def write_performance(
    config,
    meta_iteration,
    cumulative_performance,
    performance,
    step,
    num_mutated_tasks,
    num_policy_tasks,
):
    results_dir = config.run_dir + "/performance.csv"
    with open(results_dir, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                meta_iteration,
                cumulative_performance,
                performance,
                step,
                num_mutated_tasks,
                num_policy_tasks,
            ]
        )


def get_num_programs(agent, mode="policy"):
    return copy.copy(len(agent.replay_buffer.programs[mode]))


def get_num_tasks(agent, mode="policy"):
    return copy.copy(len(agent.replay_buffer.entries[mode]))


def filter_and_load_mutated_tasks(mutated_train_tasks):
    filtered_mutated_tasks = {}
    for task in mutated_train_tasks.values():
        too_big = False
        for training_example in task["training_examples"]:
            if get_num_pixels(training_example["output"]) > 1_000:
                too_big = True
        if not too_big:
            filtered_mutated_tasks[task["task_key"]] = from_dict(task)
    print(f"removed {len(mutated_train_tasks)-len(filtered_mutated_tasks)} tasks from genetic!!!")
    return filtered_mutated_tasks


@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(config: Any) -> None:

    print("\n" + "=" * 10, "Configuration", "=" * 10)

    pl.seed_everything(config.seed)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    config.trainer.profiler = None
    agent = Agent(config=config)

    

    logger = TensorBoardLogger(
        save_dir=config.run_dir, version=None, name="tensorboard", default_hp_metric=False
    )

    pl_module = HFModule(config)
    checkpoint_callback = HfModelCheckpoint(
        dirpath=f"{config.model.models_dir}/",
        save_top_k=0,
        save_last=config.model.save_last,
        config=config,
    )
    callbacks = [checkpoint_callback]

    print("initialising trainer")
    trainer = pl.Trainer(**config.trainer, logger=logger, callbacks=callbacks)

    # load tasks for ablations
    if config.ablation.used:
        print("preparing ablations")
        assert config.exit.add_policy_samples == False
        print(
            f"sample values from {config.ablation.start_value} to {config.ablation.final_value} at interval {config.ablation.mutation_interval}"
        )
        sample_values = list(
            range(
                config.ablation.start_value,
                config.ablation.final_value + config.ablation.mutation_interval,
                config.ablation.mutation_interval,
            )
        )
        mutated_tasks_files = [
            config.ablation.tasks_file + "_" + str(value) + ".json" for value in sample_values
        ]
        mutated_tasks_list = []
        for file in mutated_tasks_files:
            print(f"loading file {file}")
            with open(file, "r") as f:
                mutated_tasks = json.load(f)
            print(f"loaded {len(mutated_tasks)} tasks")
            print(f"filtering tasks from file {file}")
            mutated_tasks = filter_and_load_mutated_tasks(mutated_tasks)
            if not config.final_experiments:
                print(f"filtering out validation keys from tasks from file {file}")
                mutated_tasks = filter_by_inference_keys(
                    mutated_tasks, agent.inference_tasks.keys()
                )
            print(f"adding {len(mutated_tasks)} tasks to task list")
            mutated_tasks_list.append(mutated_tasks)
        if len(mutated_tasks_list) != config.exit.n_iters:
            raise Exception(
                f"mutated task list length: {len(mutated_tasks_list)} num iters: {config.exit.n_iters}"
            )

  

    for n_iter in range(0, config.exit.n_iters):

        if config.profile:
            profiler = cProfile.Profile()
            profiler.enable()

        # add mutated tasks to buffer
        if config.ablation.used:
            agent.add_tasks_to_buffer(
                tasks=mutated_tasks_list[n_iter], iteration_id=n_iter, mode="mutated"
            )

        data_module = ExItDataModule(config=config, replay_buffer=agent.replay_buffer)
        data_module.setup()

        # train
        t = time.time()
        trainer.fit(pl_module, datamodule=data_module)
        trainer.logger.log_metrics({"train_time": time.time() - t}, trainer.global_step)

        torch.cuda.empty_cache()
        gc.collect()
        trainer.fit_loop.max_epochs += config.trainer.max_epochs


    return trainer, data_module, pl_module


if __name__ == "__main__":
    main()
