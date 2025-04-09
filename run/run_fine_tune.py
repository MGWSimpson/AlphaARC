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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from lightning.pytorch.callbacks import Callback

class LossLoggerCallback(Callback):
    def __init__(self, save_dir="loss_logs", filename="loss_log.csv"):
        super().__init__()
        self.save_dir = save_dir
        self.filename = filename
        self.filepath = os.path.join(self.save_dir, self.filename)

        # Ensure the directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        # Create and initialize the CSV file if it doesn't exist
        if not os.path.isfile(self.filepath):
            with open(self.filepath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "global_step", "train_loss", "val_loss"])

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss", None)
        val_loss = metrics.get("val_loss", None)

        train_loss_val = train_loss.item() if train_loss is not None else None
        val_loss_val = val_loss.item() if val_loss is not None else None

        # Log to TensorBoard
        if train_loss_val is not None:
            trainer.logger.log_metrics({"train_loss_epoch": train_loss_val}, step=trainer.global_step)
        if val_loss_val is not None:
            trainer.logger.log_metrics({"val_loss_epoch": val_loss_val}, step=trainer.global_step)

        # Print to console
        print(f"[Epoch {trainer.current_epoch}] Train Loss: {train_loss_val} | Val Loss: {val_loss_val}")

        # Append to CSV
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trainer.current_epoch,
                trainer.global_step,
                train_loss_val,
                val_loss_val
            ])
def filter_by_inference_keys(tasks, inference_keys):
    task_keys = [
        s for s in tasks.keys() if not any(s.startswith(prefix) for prefix in inference_keys)
    ]
    filtered_tasks = {}
    for key in task_keys:
        filtered_tasks[key] = tasks[key]
    return filtered_tasks




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
    callbacks = [checkpoint_callback, LossLoggerCallback(save_dir=config.run_dir)]

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
