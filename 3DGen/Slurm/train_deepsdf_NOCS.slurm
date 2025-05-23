#!/bin/bash

#SBATCH -A tya@v100
#SBATCH --partition gpu_p2
#SBATCH --job-name=train_deepsdf # nom du job
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:8 # nbr de GPU par nœud
#SBATCH --exclusive
#SBATCH --cpus-per-task=24 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=100:00:00 # temps d'execution max
#SBATCH --output=log/log_deep_sdf_%j.out # fichier sortie
#SBATCH --error=log/log_error_deep_sdf_%j.out # fichier d'erreur

# nettoyage des modules charges en interactif et herites par defaut

module purge
# chargement des modules

module load singularity

#singularity exec --nv --writable-tmpfs --no-home --cleanenv   --bind $SCRATCH/volume_commun:/home/input_data/volume_commun --bind $SCRATCH/Final_NOCS/data_deepsdf/:/home/input_data/data_deepsdf --bind $SCRATCH/Final_NOCS/obj_models/:/home/input_data/obj_models   --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "
#    cd /home/input_data/volume_commun/data_to_train/deep_sdf_code_test/DeepSDF/ &&
#    source /root/miniconda3/etc/profile.d/conda.sh &&
#    conda activate deep_sdf &&
#    python -c 'import sys; print(\"Python version:\", sys.version)' &&
#    conda list &&
#    python train_deep_sdf_sup_con.py -e /home/input_data/volume_commun/data_to_train/experiement_example_NOCS/ -c latest
#  "

singularity exec --nv --writable-tmpfs --no-home --cleanenv   --bind $SCRATCH/volume_commun:/home/input_data/volume_commun --bind $SCRATCH/volume_commun/data_to_train/NOCS_origin/shapo_new/data/data_deepsdf/:/home/input_data/data_deepsdf --bind $SCRATCH/volume_commun/data_to_train/NOCS_origin/shapo_new/data/obj_models:/home/input_data/obj_models   --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "
    cd /home/input_data/volume_commun/data_to_train/deep_sdf_code_test/DeepSDF/ &&
    source /root/miniconda3/etc/profile.d/conda.sh &&
    conda activate deep_sdf &&
    python -c 'import sys; print(\"Python version:\", sys.version)' &&
    conda list &&
    python train_deep_sdf_sup_con.py -e /home/input_data/volume_commun/data_to_train/experiement_example_NOCS/ -c latest
  "






