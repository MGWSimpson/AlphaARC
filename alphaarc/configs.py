from dataclasses import dataclass
import yaml
from alphaarc.networks import BaseNetwork, PolicyValueNetwork
from alphaarc.policies import BasePolicy
from alphaarc.train import BaseTrainer, JointTrainer
from alphaarc.curriculum import BaseCurriculum
from alphaarc.env import BaseEnv, LineLevelArcEnv
from alphaarc.policies import BasePolicy, AlphaZero

@dataclass
class AlphaARCConfig:
    seed: int = 0
    n_tree_workers: int = 4
    n_examples: int = 10
    train_every: int = 10
    max_task_len: int = 512
    max_state_len:  int = 1024
    n_actions:  int = 5
    n_episodes_per_task: int = 1
    trajectory_buffer_capacity: int = 100_000
    replay_buffer_capacity: int = 100_000
    n_epochs: int = 1





def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config




# would take in some network or something then
def build_network(model_config: dict) -> BaseNetwork:
    NETWORK_REGISTRY = {"PolicyValueNetwork": PolicyValueNetwork }
    network_type = model_config['type']
    params = model_config.get('params', {})

    network_cls = NETWORK_REGISTRY.get(network_type)
    if network_cls is None:
        raise ValueError(f"Unknown network type '{network_type}'")

    return network_cls(**params)



def build_policy(policy_config: dict) -> BasePolicy:
    POLICY_REGISTRY = {"AlphaZero": AlphaZero }

    env_type = policy_config['type']
    params = policy_config.get('params', {})
    env_cls = POLICY_REGISTRY.get(env_type)
    if env_cls is None:
        raise ValueError(f"Unknown policy type '{env_type}'")

    return env_cls(**params)

def build_trainer(trainer_config: dict) -> BaseTrainer: 
    TRAINER_REGISTRY = {""}


def build_curriculum(curriculum_config: dict)-> BaseCurriculum: 
    CURRICULUM_REGISTRY = {""}

def build_env(env_config: dict) -> BaseEnv: 
    ENV_REGISTRY = {"LineLevelArcEnv": LineLevelArcEnv}

    env_type = env_config['type']
    params = env_config.get('params', {})
    env_cls = ENV_REGISTRY.get(env_type)
    if env_cls is None:
        raise ValueError(f"Unknown env type '{env_type}'")

    return env_cls(**params)

def build_alpha_arc_config(config_dict: dict) -> AlphaARCConfig:
    return AlphaARCConfig(** config_dict)