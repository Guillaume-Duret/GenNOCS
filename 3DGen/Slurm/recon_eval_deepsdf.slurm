#!/bin/bash

#SBATCH -A tya@v100
#SBfezfezfATCH --partition gpu_p2
#SBATCH --job-name=eval_deepsdf # nom du job
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:1 # nbr de GPU par nœud
#SBAezfzefzefeTCH --exclusive
#SBATCH --cpus-per-task=10 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=20:00:00 # temps d'execution max
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

full_obj="$1/obj_models/"

singularity exec --nv --writable-tmpfs --no-home --cleanenv --bind $SCRATCH/My_DeepSDF/NOCS_origin/$2:/home/input_data/$2  --bind $3:/home/input_data/deep_sdf_code_test --bind $full_obj/:/home/input_data/obj_models --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "
    cd /home/input_data/deep_sdf_code_test/DeepSDF/ &&
    source /root/miniconda3/etc/profile.d/conda.sh &&
    conda activate deep_sdf &&
    python reconstruct.py -e /home/input_data/$2 -c $4 --split /home/input_data/obj_models/split/$5.json -d /home/input_data/obj_models/data_deepsdf/ --skip
    python evaluate.py -e /home/input_data/$2 -c $4 --split /home/input_data/obj_models/split/$5.json -d /home/input_data/obj_models/data_deepsdf/
  "

#python reconstruct.py -e /home/input_data/volume_commun/data_to_train/experiement_example_NOCS/ -c 5000 --split /home/input_data/obj_models/split/val.json -d /home/input_data/data_deepsdf/ --skip

#python evaluate.py -e /home/input_data/volume_commun/data_to_train/experiement_example_NOCS/ -c 5000 --split /home/input_data/obj_models/split/val.json -d /home/input_data/obj_models/data_deepsdf/





