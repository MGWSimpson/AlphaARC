# AlphaARC

**Synthetic task generation & experiments for the Abstraction and Reasoning Corpus (ARC).**
This repository contains code and experiment notebooks for generating synthetic ARC-like tasks, running search/policy learning over those tasks, and fine-tuning models used in related experiments. The core implementation lives under `alphaarc/`; data and experiment scripts are provided in `data/`, `experiment_scripts/`, and `experiments/`.

---

## Quick overview

* Purpose: provide a procedural / programmatic toolkit to generate ARC-style tasks, run search-based and policy-learning approaches, and run experiments (including fine-tuning).
* Main places to look:

  * `alphaarc/` — primary code (search, policy learning, task generation).
  * `alphaarc/compress.py` — task / dataset generator (synthetic ARC tasks).
  * `alphaarc/search.py` and `alphaarc/policy_learning.py` — main search and policy-learning drivers.
  * `run_fine_tune.py` and `experiment_scripts/` — examples of training / experiment flows.
  * `data/` — stored datasets (mutated/generated / codeits).
  * `results/` — experiment results and finetune checkpoints.

---

## Quick start

1. Clone the repo

```bash
git clone https://github.com/MGWSimpson/AlphaARC.git
cd AlphaARC
```

2. Create environment (conda) or pip install:

```bash
# conda
conda env create -f environment.yml -n alphaarc
conda activate alphaarc

# or pip
pip install -r requirements.txt
```

The repository contains both `environment.yml` and `requirements.txt` for environment setup. A `Dockerfile` is also provided if you prefer containerized runs.

---

## Typical workflows & example commands

> These examples are intentionally high-level — many scripts accept extra config/flags. Inspect the headers / docstrings of each script for full CLI options.

### Generate synthetic tasks

Generate / compress tasks (task generator):

```bash
python alphaarc/compress.py   # generates synthetic ARC-like tasks into `data/` by default
```

### Run search / baseline experiments

Run search or mutation baselines (example):

```bash
python alphaarc/search.py     # run search-based solver over tasks
# or run any provided example scripts
bash experiment_scripts/search_1.sh
```

### Policy learning / RL-style training

Run policy learning experiments:

```bash
python alphaarc/policy_learning.py
```

### Fine-tuning a model

The repository contains `run_fine_tune.py` and finetune checkpoints in `results/finetune-checkpoints/` — use `run_fine_tune.py` to reproduce finetuning experiments or to adapt checkpoints to new generated tasks:

```bash
python run_fine_tune.py --config <your-config>
```

### Reproducing experiments

Look in `experiment_scripts/` and `experiments/` for runnable examples (shell scripts and notebooks) that demonstrate common experiment pipelines and parameter sweeps.

---

## Repository layout (high level)

```
AlphaARC/
├─ alphaarc/                # main python modules (search, policy learning, compress/task-generator, utils)
├─ data/                    # generated datasets and raw inputs
├─ experiment_scripts/      # convenience shell scripts for common experiments
├─ experiments/             # notebooks & experimental analysis
├─ results/                 # saved outputs, logs, and finetune-checkpoints
├─ environment.yml          # conda environment
├─ requirements.txt         # pip requirements
├─ Dockerfile               # optional container
├─ README.MD
└─ LICENSE                 # BSD-3-Clause-Clear
```

See the repository root for the exact file names and structure.

---

## Notes, tips & pointers

* If you’re exploring the code, start with `alphaarc/search.py` and `alphaarc/policy_learning.py` — these are the main algorithmic entry points. The generator (`compress.py`) creates synthetic tasks used in many experiments.
* Many scripts expect data to be in `data/` and will save outputs to `results/`; check and set the working paths in script arguments before running large experiments.
* Some files and notebooks may be legacy or used for quick experimentation — look for notes/headers within those files and prefer scripts under `experiment_scripts/` for reproducible runs.

---

## About ARC (context)

This repository targets the Abstraction and Reasoning Corpus (ARC) — a small-sample, compositional visual reasoning benchmark introduced to probe generalization and human-like problem solving. If you’re new to ARC, see the ARC home/info pages and the original challenge resources for background and dataset format.

---

## Citation

* If you use this code in work, please cite the paper:

Decomposing ARC Programs to Create Simpler Tasks, Matthew Simpson and Soumya Banerjee, 2026

https://www.researchgate.net/publication/399686431_Decomposing_ARC_Programs_to_Create_Simpler_Tasks



---

## Contributing

* Issues & PRs: please open an issue for discussion before major changes.
* Code style: follow existing patterns in the repo; add tests or notebooks to demonstrate new functionality.
* When adding experiments, provide a matching script under `experiment_scripts/` so others can reproduce runs.

---

## Contact / further help

* Repository owner on GitHub: `MGWSimpson` — open an issue or PR on GitHub for questions and suggested improvements.


* basically the main files you need too look at are in alpharc are policy_learning.py and search.py aswell the compress.py is the task generator and the GRPO file is just that. and the run_fine_tune.py everything else is fairly auxillary / good chance of being old.


* you can see examples of how to run things in the experiments script.


* all the files in data are codeits

* all results in results are mine + the finetune-checkpoints are where youll find the final models used. Also the finetune stuff is where you will see i tried a bunch of params.
