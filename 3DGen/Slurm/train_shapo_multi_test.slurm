#!/bin/bash

#SBATCH -A tya@v100
#SBATCH --partition gpu_p2
#SBAfefzzefeTCH -C v100-32g
#SBATCH --job-name=train_shapo # nom du job
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=8 #8 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:8 #8 # nbr de GPU par nœud
#SBATCH --exclusive
#SBATCH --cpus-per-task=3 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=40:00:00 # temps d'execution max
#SBATCH --output=log/log_shapo_test_sdf_%j.out # fichier sortie
#SBATCH --error=log/log_error_shapo_test_sdf_%j.out # fichier d'erreur

# nettoyage des modules charges en interactif et herites par defaut

#module purge
# chargement des modules


export OMP_NUM_THREADS=3
export MASTER_ADDR=$(hostname)   
export MASTER_PORT=52692
export WORLD_SIZE=8 #8   
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_LOCALID
export NODE_RANK=0

echo "Task Info:"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "SLURM_LOCALID=$SLURM_LOCALID"
echo "LOCAL_RANK=$LOCAL_RANK"
echo "RANK=$RANK"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"



module load pytorch-gpu/py3/1.8.0
conda activate Shapo2_new # Shapo2
cd /lustre/fsn1/projects/rech/tya/ubn15wo/volume_commun/data_to_train/NOCS_origin/shapo_new
srun ./ult_runner.sh $1 

#net_train.py @configs/net_config.txt



