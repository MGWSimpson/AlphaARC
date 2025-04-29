from dataclasses import dataclass
import yaml

@dataclass
class RLTrainingConfig:
    rl_batch_size: int =1
    rl_lr: float= 5e-5

@dataclass
class SupervisedTrainingConfig:
    supervised_batch_size: int = 1
    supervised_lr: float = 5e-5

@dataclass
class ModelConfig:
    model_path: str = 'finetune/2025-04-18_12-38-42/model'
    tokenizer_path: str = 'Salesforce/codet5p-220m'
    model_temperature: float = 0.95
    device: str = 'cuda'


@dataclass
class AlphaARCConfig:
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
    train_every: int = 10
    n_epochs: int = 1
    evaluation_action_temperature: float = 0
    n_tree_workers: int = 4



def load_config(path: str) -> AlphaARCConfig:
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return AlphaARCConfig(**cfg_dict)



def override_config(config: AlphaARCConfig, args): 
    raise NotImplementedError