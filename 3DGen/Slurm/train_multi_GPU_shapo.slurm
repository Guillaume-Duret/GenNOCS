#!/bin/bash
#SBATCH --job-name=multi_gpu_training       # Job name
#SBATCH --output=logs/%x-%j.out             # Output log file
#SBATCH --time=02:00:00                    # Max runtime
#SBATCH --qos=qos_gpu-dev                  # Specify QOS for GPU jobs
#SBATCH --nodes=1                          # Use one node
#SBATCH --ntasks-per-node=2                # Two tasks per node (1 task per GPU)
#SBATCH --cpus-per-task=20                 # Number of CPU cores per task
#SBATCH --gres=gpu:2                       # Use 2 GPUs
#SBATCH --hint=multithread                 # Hint to use multi-threading

# Load PyTorch module
module load pytorch-gpu/py3/1.8.0

# Activate conda environment
conda activate Shapo2

# Navigate to your project directory
cd /lustre/fsn1/projects/rech/tya/ubn15wo/volume_commun/data_to_train/NOCS_origin/shapo

# Ensure necessary environment variables are set for multi-GPU training
export MASTER_ADDR=$(hostname)               # Set master node address (hostname of this machine)
export MASTER_PORT=29500                     # Set a port for communication (you can use any available port)
export WORLD_SIZE=2                          # Total number of processes (total GPUs, 2 in your case)
export LOCAL_RANK=$SLURM_LOCALID             # Rank of the current process on the node (use $SLURM_LOCALID for this)
export NODE_RANK=0                           # Rank of the current node (use 0 for a single-node job)
export CUDA_VISIBLE_DEVICES=0,1              # Make sure the GPUs are accessible (GPU 0 and 1)

# Run the training script with the necessary configurations
./runner.sh net_train.py @configs/net_config.txt

