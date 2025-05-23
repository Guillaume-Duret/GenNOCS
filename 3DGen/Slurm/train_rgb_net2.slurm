#!/bin/bash

#SBATCH -A tya@v100
#SBATCH -C v100-32g
#SBAezfezTCH --partition gpu_p2
#SBATCH --job-name=train_rgb # nom du job
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:1 # nbr de GPU par nœud
#SBAzfezzcTCH --exclusive
#SBATCH --cpus-per-task=10 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=100:00:00 # temps d'execution max
#SBATCH --output=log/log_rgb_sdf_%j.out # fichier sortie
#SBATCH --error=log/log_error_rgb_sdf_%j.out # fichier d'erreur

# nettoyage des modules charges en interactif et herites par defaut

module purge
# chargement des modules

module load singularity
/home/input_data

full_obj="$1/obj_models/"
sdf_path="/home/input_data/$2/LatentCodes/$4.pth"
rgb_path="/home/input_data/$2/$5_$4"

echo $sdf_path
echo $rgb_path

singularity exec --nv --writable-tmpfs --no-home --cleanenv --bind $SCRATCH/My_DeepSDF/NOCS_origin/$2:/home/input_data/$2  --bind $3:/home/input_data/deep_sdf_code_test --bind $full_obj/:/home/input_data/obj_models --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "
    cd /home/input_data/deep_sdf_code_test/DeepSDF/ &&
    source /root/miniconda3/etc/profile.d/conda.sh &&
    conda activate deep_sdf &&
    python train_rgb_net2.py --deepsdf_path $sdf_path --rgb_out_path $rgb_path
  "



#singularity exec --nv --writable-tmpfs --no-home --cleanenv   --bind $SCRATCH/volume_commun:/home/input_data/volume_commun --bind $SCRATCH/volume_commun/data_to_train/NOCS_origin/shapo_new/data/data_deepsdf/:/home/input_data/data_deepsdf --bind $SCRATCH/volume_commun/data_to_train/NOCS_origin/shapo_new/data/obj_models:/home/input_data/obj_models   --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "
#    cd /home/input_data/volume_commun/data_to_train/deep_sdf_code_test/DeepSDF/ &&
#    source /root/miniconda3/etc/profile.d/conda.sh &&
#    conda activate deep_sdf &&
#    python train_rgb_net.py
#  "

#singularity exec --nv --writable-tmpfs --no-home --cleanenv --bind $SCRATCH/volume_commun:/home/input_data/volume_commun --bind $SCRATCH/Final_NOCS/data_deepsdf/:/home/input_data/data_deepsdf --bind $SCRATCH/Final_NOCS/obj_models/:/home/input_data/obj_models --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "
#    cd /home/input_data/volume_commun/data_to_train/deep_sdf_code_test/DeepSDF/ &&
#    source /root/miniconda3/etc/profile.d/conda.sh &&
#    conda activate deep_sdf &&
#    python train_rgb_net.py
#  "


#singularity exec --nv --writable-tmpfs --no-home --cleanenv   --bind $SCRATCH/volume_commun:/home/input_data/volume_commun --bind $SCRATCH/Final_NOCS/data_deepsdf/:/home/input_data/data_deepsdf --bind $SCRATCH/Final_NOCS/obj_models/:/home/input_data/obj_models   --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "
 #   cd /home/input_data/volume_commun/data_to_train/deep_sdf_code_test/DeepSDF/ &&
 #   source /root/miniconda3/etc/profile.d/conda.sh &&
 #   conda activate deep_sdf &&
 #   python -c 'import sys; print(\"Python version:\", sys.version)' &&
 #   conda list &&
 #   python train_deep_sdf_sup_con.py -e /home/input_data/volume_commun/data_to_train/experiement_example_NOCS/ -c latest
 # "







