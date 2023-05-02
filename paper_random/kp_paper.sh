#!/bin/bash
#SBATCH -p plgrid
#SBATCH -N 1
#SBATCH --ntasks-per-node=20
#SBATCH -A plgkwantowy3-cpu
# the above lines are read and interpreted by sbatch, this one and leter are not
# command, this will be executed on a compute node
# S BATCH -o random_results_bigger_weights.txt

module load python/3.10.4-gcccore-11.3.0
# source /net/people/plgrid/plgtmek1244/QHyper/venv/bin/active

# python3 experiments.py
python3 test_random.py -w 2 --values 2 2 1 --weights 1 1 1 --samples 10000 --processes 20
