#!/bin/bash

#SBATCH -A tya@v100
#SBATCH -C v100-32g
#SBATCH --job-name=gen_img # nom du job
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:1 # nbr de GPU par nœud
#SBATCH --cpus-per-task=10 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=19:00:00 # temps d'execution max
#SBATCH --output=log/log_gen_img%j.out # fichier sortie
#SBATCH --error=log/log_error_gen_img%j.out # fichier d'erreur

# nettoyage des modules charges en interactif et herites par defaut

module purge
# chargement des modules

module load pytorch-gpu/py3/2.1.1

cd $WORK/3D_gen/3DMeshes_Generation/

time python no_prompting_image_generation.py
