import os
import subprocess
import argparse

# Path to the directory containing your mesh files

# Create the parser
parser = argparse.ArgumentParser(description="Process mesh directory and output folder.")

# Add arguments
parser.add_argument(
    "--input_path",
    type=str,
    required=True,
    help="Path to the directory containing the input"
)

# Parse the arguments
args = parser.parse_args()

# Define source and destination directories
img_dir = args.input_path + '/images' # 'output_test_NOCS_data'
mask_dir = args.input_path + '/masks' # 'output_test_NOCS_data/ZITS_inpainting/masks'  
mask_dir_mid = args.input_path + '/masks_mid' # 'output_test_NOCS_data/ZITS_inpainting/masks'  
save_path = args.input_path + '/output' # 'output_test_NOCS_data/ZITS_inpainting/images'  
save_path2 = f"{save_path}_mid"

# Ensure destination directories exist
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_path2, exist_ok=True)

# Define the paths
ckpt_path = './ckpt'
config_file = './config_list/config_ZITS_HR_places2.yml'
gpu_ids = '0'

# List all image files in the image directory
img_files = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
mask_files = sorted([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))])
mask_files_mid = sorted([f for f in os.listdir(mask_dir_mid) if os.path.isfile(os.path.join(mask_dir_mid, f))])

# Ensure there is a matching mask for each image
if len(img_files) != len(mask_files):
    print("Error: The number of images and masks do not match.")
    exit(1)

# Ensure there is a matching mask for each image
if len(img_files) != len(mask_files_mid):
    print("Error: The number of images and masks do not match.")
    exit(1)

# Iterate over the image files and run the command for each pair
for img_file, mask_file_mid in zip(img_files, mask_files_mid):
    img_path = os.path.join(img_dir, img_file)
    mask_path_mid = os.path.join(mask_dir_mid, mask_file_mid)

    command = f"python single_image_test.py --path {ckpt_path} --config_file {config_file} --GPU_ids '{gpu_ids}' --img_path {img_path} --mask_path {mask_path_mid} --save_path {save_path2}"

    # Run the command
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)

# Iterate over the image files and run the command for each pair
for img_file, mask_file in zip(img_files, mask_files):
    img_path = os.path.join(img_dir, img_file)
    mask_path = os.path.join(mask_dir, mask_file)
    
    print(img_path)
    print(mask_path)
    print(" to ", save_path)
    command = f"python single_image_test.py --path {ckpt_path} --config_file {config_file} --GPU_ids '{gpu_ids}' --img_path {img_path} --mask_path {mask_path} --save_path {save_path}"
    
    # Run the command
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)

