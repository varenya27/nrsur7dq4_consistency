#!/bin/bash
#SBATCH --job-name=param_space_checks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=10:00:00
#SBATCH --output=run_%j.log
#SBATCH --partition=cpu,cpu-preempt,umd-cscdr-cpu
#SBATCH --no-requeue
#SBATCH --mail-user=vupadhyaya@umassd.edu
#SBATCH --mail-type=END
#SBATCH --mem-per-cpu=2G

module load openmpi/4.1.6
source ~/miniforge3/etc/profile.d/conda.sh
conda activate igwn-py310

mpirun -np 64 python diff_param_space_batch.py --fmin 20
cat large_errors_wfgen_rank*.txt > large_errors_wfgen_20hz.txt && rm large_errors_wfgen_rank*.txt
cat large_errors_wrapper_rank*.txt > large_errors_wrapper_20hz.txt && rm large_errors_wrapper_rank*.txt

mpirun -np 64 python diff_param_space_batch.py --fmin 0
cat large_errors_wrapper_rank*.txt > large_errors_wrapper_0hz.txt && rm large_errors_wfgen_rank*.txt
cat large_errors_wfgen_rank*.txt > large_errors_wfgen_0hz.txt && rm large_errors_wrapper_rank*.txt

