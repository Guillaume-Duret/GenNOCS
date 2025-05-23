#!/bin/bash

#SBATCH -A tya@cpu
#SBATCH --job-name=train_deep # nom du job
#SBATCH --qos=qos_cpu-t3
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --cpus-per-task=3 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=20:00:00 # temps d'execution max
#SBATCH --output=log/log_process_without_img%j.out # fichier sortie
#SBATCH --error=log/log_error_without_img%j.out # fichier d'erreur
#SBATCH --array=0-96:3

# nettoyage des modules charges en interactif et herites par defaut

module purge

: <<'END'
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
time python3 image_generation.py ${SLURM_ARRAY_TASK_ID} $1 $2
#done

cd $WORK/3D_gen/InstantMesh

time python run.py configs/instant-mesh-large.yaml $2/$3/${SLURM_ARRAY_TASK_ID}  --save_video --export_texmap --output_path $2/$3/meshes/

module load pytorch-gpu/py3/1.9.0
conda activate blenderproc2

export GIT_PYTHON_REFRESH=quiet

cd $WORK/3D_gen/BlenderProc/blenderproc/scripts/

$mesh_path="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/meshes/"
$second_output="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/process_intermediate/"
$final_output="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/processed_meshes/"

python -m blenderproc run select_poly_all.py  --mesh_dir "$mesh_path" --destination_folder "$second_output"

cd /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/3DMeshes_Generation
python rename_file.py --folder_path "$second_output"
python run.py --input_path "$second_output" --output_path "$second_output"
python copy_to_zits.py --input_path "$second_output" --output_path "$second_output/ZITS_inpainting"

cd /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/ZITS_inpainting
python single_image_test_script.py --input_path "$second_output/ZITS_inpainting"

cd /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/3DMeshes_Generation/
python replace.py --input_dir_original "$second_output" --input_dir_inpainted "$second_output/ZITS_inpainting/output" --input_dir_mask "$second_output" --output_dir "$second_output"

cd $WORK/3D_gen/BlenderProc/blenderproc/scripts/
python -m blenderproc run update_json.py --input_path "$second_output" --output_path "$final_output"
python -m blenderproc run modify_uv_all.py --input_path "$second_output" --output_path "$final_output" --json_path "$final_output"

END


module load pytorch-gpu/py3/1.9.0
conda activate blenderproc2

export GIT_PYTHON_REFRESH=quiet

cd "$WORK/3D_gen/BlenderProc/blenderproc/scripts/"

# Corrected variable assignments
mesh_path="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/meshes/"
second_output="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/process_intermediate/"
json_output="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/processed_meshes/"

clean_output="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/clean_meshes/"
center_output="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/center_resized/"
aligned_output="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/aligned_meshes/"
aligned_output2="$1/$2/meshes/${SLURM_ARRAY_TASK_ID}/instant-mesh-large/aligned_meshes2/"



# texture inpainting


# Execute the first script with corrected paths
#python -m blenderproc run select_poly_all.py --mesh_dir "$mesh_path" --destination_folder "$second_output"

# Run additional scripts with corrected paths
cd /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/3DMeshes_Generation
#python rename_file.py --folder_path "$second_output"
#python run.py --input_path "$second_output" --output_path "$second_output"
#python copy_to_zits.py --input_path "$second_output" --output_path "$second_output/ZITS_inpainting"

# Move to the ZITS inpainting directory and run the inpainting script
cd /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/ZITS_inpainting
#python single_image_test_script.py --input_path "$second_output/ZITS_inpainting"

# Return to 3DMeshes_Generation directory and execute the replace script
cd /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/3DMeshes_Generation/
#python replace.py --input_dir_original "$second_output" --input_dir_inpainted "$second_output/ZITS_inpainting/output" --input_dir_mask "$second_output" --output_dir "$second_output"

# Return to the BlenderProc scripts directory and run the final processing scripts
cd "$WORK/3D_gen/BlenderProc/blenderproc/scripts/"
#python -m blenderproc run update_json.py --input_path "$second_output" --output_path "$json_output"
python -m blenderproc run modify_uv_all.py --input_path "$second_output" --output_path "$json_output" --json_path "$json_output"

# mesh processing

cd /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/shapo/mesh_transformation/

#time python remove_artefact.py --input_folder "$mesh_path" --output_folder "$clean_output" --area_threshold 0.0001 --vertex_threshold 10 --num_to_remove 5 --proximity_threshold 0.02

#time python rescale_center_mesh.py --input_dir "$mesh_path" --out_dir "$center_output"
#time python rescale_center_mesh.py --input_dir "$center_output" --out_dir "$center_output"
time python rescale_center_mesh.py --input_dir "$json_output" --out_dir "$center_output"

time python align_mesh_good.py --input_dir "$center_output" --out_dir "$aligned_output" # --reference_path "-1"
time python postprocessing.py --original_dir "$mesh_path" --dir_to_update "$aligned_output" --out_dir "$aligned_output"

#rm -rf "$aligned_output2"
#mkdir "$aligned_output2"

#time python rescale_center_mesh.py --input_dir "$aligned_output" --out_dir "$aligned_output2" --invert_normals
#time python postprocessing.py --original_dir "$aligned_output" --dir_to_update "$aligned_output2" --out_dir "$aligned_output2"

# reorganize for final version
#     out_dir2="/data/gduret/CAT6D/NOCS_data/${bowl_type}/meshes/${i}/instant-mesh-large/final_meshes_final/"
#    out_dir_final="/data/gduret/CAT6D/NOCS_data/obj_models/"

obj_out="$1/obj_models/"

python split_data.py --input_dir "$aligned_output" --out_dir "$obj_out"





: <<'END'

obj_out="$1/obj_models/${SLURM_ARRAY_TASK_ID}"
rm -rf "$1/obj_models/${SLURM_ARRAY_TASK_ID}"
rm -rf "$obj_out/split/${SLURM_ARRAY_TASK_ID}"

mkdir "$1/obj_models/"
mkdir $obj_out
mkdir "$obj_out/split"
mkdir "$obj_out/split/${SLURM_ARRAY_TASK_ID}"

python split_data.py --input_dir "$aligned_output2" --out_dir "$obj_out"
python create_split.py --input_dir "$obj_out" --out_dir "$obj_out/split/"

# run in paralele val part

cd /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/3DMeshes_Generation/Slurm
sbatch val_cpu_without_tex_process_all_generated_meshes.slurm $1 $2 $3 ${SLURM_ARRAY_TASK_ID}


full_obj="$1/obj_models/"
# Final realign + Deepsdf process dataset 
# $3 is the run_process_script

module load singularity


singularity exec --nv --writable-tmpfs --no-home --cleanenv --bind $full_obj:/home/input_data/obj_models --bind $3:/home/input_data/script_deepsdf/ --bind $SCRATCH/volume_commun:/home/input_data/volume_commun --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "./home/input_data/script_deepsdf/run_preprocess.sh ${SLURM_ARRAY_TASK_ID} train /home/input_data/obj_models/${SLURM_ARRAY_TASK_ID}/"

#singularity exec --nv --writable-tmpfs --no-home --cleanenv --bind /lustre/fsn1/projects/rech/tya/ubn15wo/3D_GEN/Fruits_data/obj_models/:/home/input_data/obj_models --bind /lustre/fswork/projects/rech/tya/ubn15wo/3D_gen/3DMeshes_Generation/Slurm/script_deepsdf/:/home/input_data/script_deepsdf/ --bind $SCRATCH/volume_commun:/home/input_data/volume_commun   --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "./home/input_data/script_deepsdf/run_preprocess.sh 0 train /home/input_data/obj_models/0/"

#singularity exec --nv --writable-tmpfs --no-home --cleanenv --bind $obj_out:/home/input_data/ --bind $3:/home/input_data --bind $SCRATCH/volume_commun:/home/input_data/   --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "./home/input_data/run_preprocess.sh ${SLURM_ARRAY_TASK_ID} train /home/input_data/obj_models/${SLURM_ARRAY_TASK_ID}/"

#singularity exec --nv --writable-tmpfs --no-home --cleanenv --bind $obj_out:/home/input_data/ --bind $3:/home/input_data --bind $SCRATCH/volume_commun:/home/input_data/   --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "./home/input_data/run_preprocess.sh ${SLURM_ARRAY_TASK_ID} val /home/input_data/obj_models/${SLURM_ARRAY_TASK_ID}/"


#singularity exec --nv --writable-tmpfs --no-home --cleanenv   --bind $SCRATCH/volume_commun:/home/input_data   --network=host $SINGULARITY_ALLOWED_DIR/deepsdf.sif /bin/bash -c "./home/input_data/run_preprocess.sh ${SLURM_ARRAY_TASK_ID} val "

END




