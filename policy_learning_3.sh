# eval run
/home/mgws3/.conda/envs/AlphaARC/bin/python /home/mgws3/AlphaARC/alphaarc/policy_learning.py --config_path alphaarc/configs/policy_learning/sample_eval.yaml --seed 1  --n_epochs 10
/home/mgws3/.conda/envs/AlphaARC/bin/python /home/mgws3/AlphaARC/alphaarc/policy_learning.py --config_path alphaarc/configs/policy_learning/grpo_eval.yaml --seed 1 --n_epochs 10
/home/mgws3/.conda/envs/AlphaARC/bin/python /home/mgws3/AlphaARC/alphaarc/policy_learning.py --config_path alphaarc/configs/policy_learning/sparse_grpo_eval.yaml --seed 1 --n_epochs 10
/home/mgws3/.conda/envs/AlphaARC/bin/python /home/mgws3/AlphaARC/alphaarc/policy_learning.py --config_path alphaarc/configs/policy_learning/internal_grpo_eval.yaml --seed 1 --n_epochs 10
