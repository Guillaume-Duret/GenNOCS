#!/bin/bash

#SBATCH -A tya@v100
#SsvdsvdBATCH --partition gpu_p2
#SBATCH -C v100-32g
#SBATCH --job-name=train_shapo # nom du job
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:1 # nbr de GPU par nœud
#SBdfvfdvfdvfdATCH --exclusive
#SBATCH --cpus-per-task=10 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=100:00:00 # temps d'execution max
#SBATCH --output=log/log_shapo_%j.out # fichier sortie
#SBATCH --error=log/log_error_shapo_%j.out # fichier d'erreur

# nettoyage des modules charges en interactif et herites par defaut

module purge
# chargement des modules

#module load singularity

#singularity exec --nv --writable-tmpfs --no-home --cleanenv   --bind $SCRATCH/volume_commun:/home/input_data/volume_commun --bind $SCRATCH/Final_NOCS/data_deepsdf/:/home/input_data/data_deepsdf --bind $SCRATCH/Final_NOCS/obj_models/:/home/input_data/obj_models   --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "
#    cd /home/input_data/volume_commun/data_to_train/deep_sdf_code_test/DeepSDF/ &&
#    source /root/miniconda3/etc/profile.d/conda.sh &&
#    conda activate deep_sdf &&
#    python -c 'import sys; print(\"Python version:\", sys.version)' &&
#    conda list &&
#    python train_deep_sdf_sup_con.py -e /home/input_data/volume_commun/data_to_train/experiement_example_NOCS/ -c latest
#  "

module load pytorch-gpu/py3/1.8.0

conda activate Shapo2

cd /lustre/fsn1/projects/rech/tya/ubn15wo/volume_commun/data_to_train/NOCS_origin/shapo

./runner.sh net_train.py @configs/net_config.txt




