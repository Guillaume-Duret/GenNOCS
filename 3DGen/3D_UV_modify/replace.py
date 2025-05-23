import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def replace_with_original(original_image_path, inpainted_image_path, mask_image_path, output_image_path, reverse):
    # Load images
    original_img = cv2.imread(original_image_path)
    inpainted_img = cv2.imread(inpainted_image_path)
    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)  # Mask should be in grayscale

    # Check if images have been loaded correctly
    if original_img is None or inpainted_img is None or mask is None:
        print(f"Error loading images for {original_image_path}, {inpainted_image_path}, or {mask_image_path}.")
        return

    # Ensure all images have the same dimensions
    if original_img.shape != inpainted_img.shape or original_img.shape[:2] != mask.shape:
        print(f"Images and mask dimensions do not match for {original_image_path}, {inpainted_image_path}, or {mask_image_path}.")
        return

    print(f"paths : original_image_path : {original_image_path}, inpainted_image_path : {inpainted_image_path}, mask_image_path : {mask_image_path}, output_image_path : {output_image_path}")

    # Convert mask to binary (0 or 255) if it's not already
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if reverse == True : 
        binary_mask = cv2.bitwise_not(binary_mask)

    # Replace regions in the inpainted image with regions from the original image using the mask
    # Convert mask to 3 channels to match the original image
    binary_mask_colored = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    
    # Apply mask to original image and inpainted image
    region_from_original = cv2.bitwise_and(original_img, binary_mask_colored)
    region_from_inpainted = cv2.bitwise_and(inpainted_img, cv2.bitwise_not(binary_mask_colored))

    cv2.imwrite(f"{output_image_path}_orig", region_from_original)
    cv2.imwrite(f"{output_image_path}_inpa", region_from_inpainted)

    # Combine the two regions
    modified_img = cv2.add(region_from_original, region_from_inpainted)

    # Save the output image
    cv2.imwrite(output_image_path, modified_img)
    print(f"Modified image saved as {output_image_path}")

    # Convert images from BGR to RGB for matplotlib
    #original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    #inpainted_img_rgb = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)
    #modified_img_rgb = cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB)

import argparse

# Path to the directory containing your mesh files
# Create the parser
parser = argparse.ArgumentParser(description="Process mesh directory and output folder.")

# Add arguments
parser.add_argument(
    "--input_dir_original",
    type=str,
    required=True,
    help="Path to the directory containing the input"
)
parser.add_argument(
    "--input_dir_inpainted",
    type=str,
    required=True,
    help="Destination"
)
parser.add_argument(
    "--input_dir_mask",
    type=str,
    required=True,
    help="Path to the directory containing the input"
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Destination"
)

# Parse the arguments
args = parser.parse_args()

# Define input and output directories
input_dir_original = args.input_dir_original #'output_test_NOCS_data/'
input_dir_inpainted = args.input_dir_inpainted #'output_test_NOCS_data/ZITS_inpainting/output'
input_dir_inpainted2 = f"{args.input_dir_inpainted}_mid"
input_dir_mask = args.input_dir_mask #'output_test_NOCS_data/'
output_dir = args.output_dir # 'output_test_NOCS_data/'
output_dir2 = f"{args.output_dir}_2"
output_dir3 = f"{args.output_dir}_3"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

if not os.path.exists(output_dir3):
    os.makedirs(output_dir3)

# List all original image files in the input directory
original_files = sorted([f for f in os.listdir(input_dir_inpainted) if f.endswith('.png')])

# Process each file
for original_file in original_files:
    inpainted_file = original_file
    base_name = original_file.replace('_test.png', '')
    
    mask_file = f"{base_name}_filtered_out_mask.png"
    mask_file2 = f"{base_name}_mask.png" # filter_orange_gray_black.png"
    mask_file3 = f"{base_name}_mask.png"
    output_file = f"{base_name}.png"

    original_path = os.path.join(input_dir_original, original_file)
    inpainted_path = os.path.join(input_dir_inpainted, inpainted_file)
    inpainted_path2 = os.path.join(input_dir_inpainted2, inpainted_file)
    mask_path = os.path.join(input_dir_mask, mask_file)
    mask_path2 = os.path.join(input_dir_mask, mask_file2)
    mask_path3 = os.path.join(input_dir_mask, mask_file3)
    output_path = os.path.join(output_dir, output_file)
    output_path2 = os.path.join(output_dir2, output_file)
    output_path3 = os.path.join(output_dir3, output_file)

    print("mask_file : ", mask_file)
    print("output_file : ", output_file)

    replace_with_original(original_path, inpainted_path, mask_path, output_path, False)
    replace_with_original(original_path, inpainted_path, mask_path2, output_path2, True)
    replace_with_original(original_path, inpainted_path2, mask_path3, output_path3, True)
