#!/bin/bash

#SedfzefzBATCH -A uli@cpu
#SBAezfeTCH --job-name=pr_texture # nom du job
#SBzfATCH --qos=qos_cpu-t3
#SBAezfeTCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBAzefeTCH --cpus-per-task=1 # nbr de CPU par tache
#SBAzefeTCH --hint=nomultithread # pas de hyperthreading
#SBAzefeTCH --time=20:00:00 # temps d'execution max
#SBAzefeTCH --output=log/log_process_without_img%j.out # fichier sortie
#SBAzefeTCH --error=log/log_error_without_img%j.out # fichier d'erreur
#SBAzefeTCH --array=0-14:3

#SBATCH -A esp@v100
#SBATCH -C v100-32g
#SBATCH --job-name=process # nom du job
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:1 # nbr de GPU par nœud
#SBATCH --cpus-per-task=10 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=10:00:00 # temps d'execution max
#SBATCH --output=log/log_gen_img%j.out # fichier sortie
#SBATCH --error=log/log_error_gen_img%j.out # fichier d'erreur
#SBATCH --array=0-0:3


# nettoyage des modules charges en interactif et herites par defaut

module purge

module load pytorch-gpu/py3/1.9.0
conda activate blenderproc2

export GIT_PYTHON_REFRESH=quiet

cd "$WORK/3D_gen/BlenderProc/blenderproc/scripts/"

# Corrected variable assignments
mesh_path="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/meshes/"
second_output="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/process_intermediate/"
json_output="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/processed_meshes/"

#clean_output="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/clean_meshes/"
#center_output="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/center_resized/"
#aligned_output="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/aligned_meshes/"
#aligned_output2="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/aligned_meshes2/"


# texture inpainting

# Execute the first script with corrected paths

echo "--------------------select_poly_all.py--------------------------"
time python -m blenderproc run select_poly_all.py --mesh_dir "$mesh_path" --destination_folder "$second_output"

# Run additional scripts with corrected paths
cd /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/3DMeshes_Generation
echo "--------------------rename_file.py--------------------------"
time python rename_file.py --folder_path "$second_output"
echo "--------------------run.py--------------------------"
time python run.py --input_path "$second_output" --output_path "$second_output"
echo "--------------------copy_to_zits.py--------------------------"
time python copy_to_zits.py --input_path "$second_output" --output_path "$second_output/ZITS_inpainting"

# Move to the ZITS inpainting directory and run the inpainting script
#cd /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/ZITS_inpainting
echo "--------------------single_image_test_script.py--------------------------"
time python single_image_test_script.py --input_path "$second_output/ZITS_inpainting"

# Return to 3DMeshes_Generation directory and execute the replace script
#cd /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/3DMeshes_Generation/
echo "--------------------replace.py--------------------------"
#time python replace.py --input_dir_original "$second_output" --input_dir_inpainted "$second_output/ZITS_inpainting/output" --input_dir_mask "$second_output" --output_dir "$second_output"

# Return to the BlenderProc scripts directory and run the final processing scripts
#cd "$WORK/3D_gen/BlenderProc/blenderproc/scripts/"
echo "--------------------update_json.py--------------------------"
#time python -m blenderproc run update_json.py --input_path "$second_output" --output_path "$json_output"
echo "--------------------modify_uv_all.py--------------------------"
#time python -m blenderproc run modify_uv_all.py --input_path "$second_output" --output_path "$json_output" --json_path "$json_output"


# mesh processing

cd /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/shapo/mesh_transformation/

#time python remove_artefact.py --input_folder "$mesh_path" --output_folder "$clean_output" --area_threshold 0.0001 --vertex_threshold 10 --num_to_remove 5 --proximity_threshold 0.02

#time python rescale_center_mesh.py --input_dir "$mesh_path" --out_dir "$center_output"
#time python rescale_center_mesh.py --input_dir "$center_output" --out_dir "$center_output"
#time python rescale_center_mesh.py --input_dir "$json_output" --out_dir "$center_output"

#time python align_mesh.py --input_dir "$center_output" --out_dir "$aligned_output"
#time python postprocessing.py --original_dir "$mesh_path" --dir_to_update "$aligned_output"

#rm -rf "$aligned_output2"
#mkdir "$aligned_output2"

#time python rescale_center_mesh.py --input_dir "$aligned_output" --out_dir "$aligned_output2" --invert_normals
#time python postprocessing.py --original_dir "$aligned_output" --dir_to_update "$aligned_output2" --out_dir "$aligned_output2"

# reorganize for final version


#     out_dir2="/data/gduret/CAT6D/NOCS_data/${bowl_type}/meshes/${i}/instant-mesh-large/final_meshes_final/"
#    out_dir_final="/data/gduret/CAT6D/NOCS_data/obj_models/"



obj_out="$1/$2/obj_models/${SLURM_ARRAY_TASK_ID}"
#rm -rf "$1/$2/obj_models/${SLURM_ARRAY_TASK_ID}"
#rm -rf "$obj_out/split/${SLURM_ARRAY_TASK_ID}"

#mkdir "$1/obj_models/"
#mkdir $obj_out
#mkdir "$obj_out/split"
#mkdir "$obj_out/split/${SLURM_ARRAY_TASK_ID}"

#python split_data.py --input_dir "$aligned_output" --out_dir "$obj_out"
#python create_split.py --input_dir "$obj_out" --out_dir "$obj_out/split/${SLURM_ARRAY_TASK_ID}"

# run in paralele val part

cd /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/3DMeshes_Generation/Slurm
#sbatch val_cpu_without_tex_process_all_generated_meshes.slurm $1 $2 $3 ${SLURM_ARRAY_TASK_ID}


full_obj="$1/obj_models/"
# Final realign + Deepsdf process dataset 
# $3 is the run_process_script

#module load singularity


#singularity exec --nv --writable-tmpfs --no-home --cleanenv --bind $full_obj:/home/input_data/obj_models --bind $3:/home/input_data/script_deepsdf/ --bind $SCRATCH/volume_commun:/home/input_data/volume_commun --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "./home/input_data/script_deepsdf/run_preprocess.sh $4 train /home/input_data/obj_models/"

#singularity exec --nv --writable-tmpfs --no-home --cleanenv --bind $full_obj:/home/input_data/obj_models --bind $3:/home/input_data/script_deepsdf/ --bind $SCRATCH/volume_commun:/home/input_data/volume_commun --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "./home/input_data/script_deepsdf/run_preprocess.sh 99 val /home/input_data/obj_models/"



#singularity exec --nv --writable-tmpfs --no-home --cleanenv --bind /lustre/fsn1/projects/rech/tya/ubn15wo/3D_GEN/Fruits_data/obj_models/:/home/input_data/obj_models --bind /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/3DMeshes_Generation/Slurm/script_deepsdf/:/home/input_data/script_deepsdf/ --bind $SCRATCH/volume_commun:/home/input_data/volume_commun   --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "./home/input_data/script_deepsdf/run_preprocess.sh 0 train /home/input_data/obj_models/0/"

#singularity exec --nv --writable-tmpfs --no-home --cleanenv --bind $obj_out:/home/input_data/ --bind $3:/home/input_data --bind $SCRATCH/volume_commun:/home/input_data/   --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "./home/input_data/run_preprocess.sh ${SLURM_ARRAY_TASK_ID} train /home/input_data/obj_models/${SLURM_ARRAY_TASK_ID}/"

#singularity exec --nv --writable-tmpfs --no-home --cleanenv --bind $obj_out:/home/input_data/ --bind $3:/home/input_data --bind $SCRATCH/volume_commun:/home/input_data/   --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "./home/input_data/run_preprocess.sh ${SLURM_ARRAY_TASK_ID} val /home/input_data/obj_models/${SLURM_ARRAY_TASK_ID}/"


#singularity exec --nv --writable-tmpfs --no-home --cleanenv   --bind $SCRATCH/volume_commun:/home/input_data   --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "./home/input_data/run_preprocess.sh ${SLURM_ARRAY_TASK_ID} val "






