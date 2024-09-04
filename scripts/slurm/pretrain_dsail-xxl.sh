#!/bin/bash
#SBATCH --job-name=dsail-xxl               # Name of the job, used in job scheduling and output filenames
#SBATCH --output=logs/%x_%j.log            # File to which stdout and stderr will be written; %x is job name, %j is job ID
#SBATCH --open-mode=append                 # Append to the output file instead of overwriting it
#SBATCH --partition=snlp                   # Specify the partition/queue to submit the job to
#SBATCH --nodes=1                          # Number of nodes to allocate for the job
#SBATCH --ntasks-per-node=4                # Number of tasks to run per node
#SBATCH --cpus-per-task=24                 # Number of CPUs allocated per task
#SBATCH --gres=gpu:4                       # Number of GPUs to allocate (4 GPUs in this case)

# Initialize module
source /etc/profile.d/modules.sh

# Load modules
module load cuda/cuda-12.3
module load mpi/openmpi-5.0.3

# Initialize conda
__conda_setup="$($HOME/miniconda3/bin/conda shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# Activate conda environment
conda activate litgpt

# Run scripts
srun litgpt pretrain --config config_hub/pretrain/dsail-xxl.yaml
