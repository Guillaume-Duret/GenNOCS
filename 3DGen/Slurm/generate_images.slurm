#!/bin/bash

#SBATCH -A tya@v100
#SBATCH -C v100-32g
#SBATCH --job-name=gen_img # nom du job
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:1 # nbr de GPU par nœud
#SBATCH --cpus-per-task=10 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=20:00:00 # temps d'execution max
#SBATCH --output=log/log_gen_img%j.out # fichier sortie
#SBATCH --error=log/log_error_gen_img%j.out # fichier d'erreur
#SBATCH --array=0-100:1

# nettoyage des modules charges en interactif et herites par defaut

module purge
# chargement des modules

module load pytorch-gpu/py3/2.1.1

cd $WORK/3D_gen/3DMeshes_Generation/


# Path to the prompt file
PROMPT_FILE=$1

echo $PROMPT_FILE

# Check if the file exists
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Error: $PROMPT_FILE not found!"
    exit 1
fi

# Get the total number of prompts (lines in the file)
TOTAL_PROMPTS=$(wc -l < "$PROMPT_FILE")

echo $TOTAL_PROMPTS

# Iterate over each prompt ID (starting from 0)
#for (( i=0; i<$TOTAL_PROMPTS; i++ ))
#do
echo "Running image generation for prompt ID: $i"
python3 image_generation.py ${SLURM_ARRAY_TASK_ID} $1
#done


#time python image_generation.py
