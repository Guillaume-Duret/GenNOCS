#!/bin/bash

#SBATCH -A tya@cpu
#SBATCH --job-name=train_deep # nom du job
#SBATCH --qos=qos_cpu-t3
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --cpus-per-task=2 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=20:00:00 # temps d'execution max
#SBATCH --output=log/cp%j.out # fichier sortie
#SBATCH --error=log/cp_error%j.out # fichier d'erreur
#SregregBATCH --array=0-6:3

# nettoyage des modules charges en interactif et herites par defaut


cp -r /lustre/fsn1/projects/rech/tya/ubn15wo/3D_GEN/Vegetables_data/obj_models /lustre/fsn1/projects/rech/tya/ubn15wo/3D_GEN/Full_vegetables_fruits/


