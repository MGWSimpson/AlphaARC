from dataclasses import dataclass
import yaml
from alphaarc.networks import BaseNetwork, PolicyValueNetwork, ActionNetwork, PolicyNetwork, AlphaZeroNetwork
from alphaarc.train import BaseTrainer, JointTrainer
from alphaarc.curriculum import BaseCurriculum
from alphaarc.env import BaseEnv, LineLevelArcEnv
from alphaarc.policies import BasePolicy, AlphaProofPolicy, MCTSPolicy, PolicyGuidedMCTSPolicy, AlphaZeroPolicy
from alphaarc.curriculum import BaseCurriculum, BaselineCurriculum


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
    NETWORK_REGISTRY = {"PolicyValueNetwork": PolicyValueNetwork, 
                        "ActionNetwork": ActionNetwork, 
                        "PolicyNetwork": PolicyNetwork,
                        "AlphaZeroNetwork": AlphaZeroNetwork}
    
    network_type = model_config['type']
    params = model_config.get('params', {})

    network_cls = NETWORK_REGISTRY.get(network_type)
    if network_cls is None:
        raise ValueError(f"Unknown network type '{network_type}'")

    return network_cls(**params)



def build_policy(model, env,policy_config: dict) -> BasePolicy:
    POLICY_REGISTRY = {"AlphaProofPolicy": AlphaProofPolicy, 
                       "MCTSPolicy": MCTSPolicy,
                       "PolicyGuidedMCTSPolicy": PolicyGuidedMCTSPolicy, 
                       "AlphaZeroPolicy": AlphaZeroPolicy}

    env_type = policy_config['type']
    params = policy_config.get('params', {})
    env_cls = POLICY_REGISTRY.get(env_type)
    if env_cls is None:
        raise ValueError(f"Unknown policy type '{env_type}'")

    return env_cls(model, env, **params)

def build_trainer(trainer_config: dict) -> BaseTrainer: 
    TRAINER_REGISTRY = {"JointTrainer": JointTrainer}

    trainer_type = trainer_config['type']
    params = trainer_config.get('params', {})
    env_cls = TRAINER_REGISTRY.get(trainer_type)
    if env_cls is None:
        raise ValueError(f"Unknown curriculum type '{trainer_type}'")

    return env_cls(**params)

def build_curriculum(curriculum_config: dict)-> BaseCurriculum: 
    CURRICULUM_REGISTRY = {"BaselineCurriculum": BaselineCurriculum}

    curriculum_type = curriculum_config['type']
    params = curriculum_config.get('params', {})
    env_cls = CURRICULUM_REGISTRY.get(curriculum_type)
    if env_cls is None:
        raise ValueError(f"Unknown curriculum type '{curriculum_type}'")

    return env_cls(**params)



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