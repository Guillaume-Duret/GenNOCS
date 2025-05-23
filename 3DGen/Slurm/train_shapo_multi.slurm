#!/bin/bash

#SBATCH -A tya@v100
#SBATCH --partition gpu_p2
#SBATCH --job-name=train_shapo # nom du job
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:8 # nbr de GPU par nœud
#SBATCH --exclusive
#SBATCH --cpus-per-task=24 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=100:00:00 # temps d'execution max
#SBATCH --output=log/log_shapo_sdf_%j.out # fichier sortie
#SBATCH --error=log/log_error_shapo_sdf_%j.out # fichier d'erreur

# nettoyage des modules charges en interactif et herites par defaut

module purge
# chargement des modules

module load pytorch-gpu/py3/1.8.0
conda activate Shapo2_new
cd /lustre/fsn1/projects/rech/tya/ubn15wo/volume_commun/data_to_train/NOCS_origin/shapo_new

./runner_multi.sh net_train.py @configs/net_config.txt



