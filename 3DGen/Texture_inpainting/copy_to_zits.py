import shutil
import os
import glob
import re  # Import regex module

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
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="Destination"
)

# Parse the arguments
args = parser.parse_args()

# Define source and destination directories
source_dir = args.input_path # 'output_test_NOCS_data'
dest_dir_black = args.output_path + '/masks' # 'output_test_NOCS_data/ZITS_inpainting/masks'  
dest_dir_black_mid = args.output_path + '/masks_mid' # 'output_test_NOCS_data/ZITS_inpainting/masks'
dest_dir_apple = args.output_path + '/images' # 'output_test_NOCS_data/ZITS_inpainting/images'  

# Ensure destination directories exist
os.makedirs(dest_dir_black, exist_ok=True)
os.makedirs(dest_dir_black_mid, exist_ok=True)
os.makedirs(dest_dir_apple, exist_ok=True)

# Define file patterns
file_pattern_black = os.path.join(source_dir, '*_filter_orange_gray_black.png')
file_pattern_black_mid = os.path.join(source_dir, '*_orange_mask_mid.png')
file_pattern_apple = os.path.join(source_dir, '*_test.png')

# Find all files matching the patterns
files_to_copy_black = glob.glob(file_pattern_black)

files_to_copy_black_mid = glob.glob(file_pattern_black_mid)

print(files_to_copy_black_mid)
files_to_copy_apple = glob.glob(file_pattern_apple)
# Function to copy files to the destination directory
def copy_files(files, dest_dir):
    for file_path in files:
        # Extract the file name from the path
        file_name = os.path.basename(file_path)
        
        # Define the destination file path
        dest_file_path = os.path.join(dest_dir, file_name)
        
        # Copy the file
        shutil.copy(file_path, dest_file_path)
        print(f'Copied {file_name} to {dest_dir}')

# Copy the files
copy_files(files_to_copy_black, dest_dir_black)
copy_files(files_to_copy_apple, dest_dir_apple)
copy_files(files_to_copy_black_mid, dest_dir_black_mid)

print('File copying completed.')
