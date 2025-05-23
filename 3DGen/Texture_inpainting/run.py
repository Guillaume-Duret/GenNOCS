import os
import time
import logging
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    filename='batch_processing.log',  # Log file
    level=logging.INFO,               # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

input_path = args.input_path #"test_NOCS_data"
# Destination folder for the exported files
output_path = args.output_path #"output_test_NOCS_data"

batch_size = 10
delay_between_batches = 1  # Delay in seconds

# List all image files
image_files = [f for f in os.listdir(input_path) if f.endswith('_test.png') or f.endswith('_test.jpg') or f.endswith('_test.jpeg')]
total_files = len(image_files)
num_batches = (total_files + batch_size - 1) // batch_size

logging.info('Batch processing started.')
logging.info(f'Total files: {total_files}')
logging.info(f'Number of batches: {num_batches}')

def find_valid_patch(image, mask, patch_size=100, max_attempts=1000):
    h, w = mask.shape
    attempts = 0
    while attempts < max_attempts:
        x, y = np.random.randint(0, w - patch_size), np.random.randint(0, h - patch_size)
        patch = mask[y:y + patch_size, x:x + patch_size]
        if cv2.countNonZero(patch) == patch_size * patch_size:
            return image[y:y + patch_size, x:x + patch_size]
        attempts += 1
    x, y = np.random.randint(0, w - patch_size), np.random.randint(0, h - patch_size)
    logging.warning("No valid patch found within the maximum number of attempts.")
    return image[y:y + patch_size, x:x + patch_size]

def convert_to_rgb1(image, is_gray=False):
    """Convert image to RGB format."""
    if is_gray:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def convert_to_rgb2(image, is_gray=False):
    """Convert image to RGB format."""
    if is_gray:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_and_plot_images(images_to_save, output_dir, original_image_name):
    """Save images and plot results."""
    # Save images
    for name, img in images_to_save.items():
        is_gray = 'mask' in name
        if name == '':
            cv2.imwrite(os.path.join(output_dir, f'{original_image_name}.png'), convert_to_rgb1(img, is_gray=is_gray))
        else:
            cv2.imwrite(os.path.join(output_dir, f'{original_image_name}_{name}.png'), convert_to_rgb1(img, is_gray=is_gray))

    # Plot the results
    plt.figure(figsize=(18, 18))
    num_images = len(images_to_save)
    for i, (name, img) in enumerate(images_to_save.items()):
        plt.subplot(5, 4, i + 1)
        plt.title(name.replace('_', ' ').capitalize())
        plt.imshow(convert_to_rgb2(img, is_gray='mask' in name))
        plt.axis('off')

    # Save and show the plot
    plot_path = os.path.join(output_dir, f'{original_image_name}_process.png')
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free up memory

def apply_kernel(image, kernel_size, mask):
    """Apply kernel to the result image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def process_image(image_file, input_dir, output_dir):
    if 1==1:
        # Define full path for the current image
        original_image_path = os.path.join(input_dir, image_file)
        
        # Extract the base name without extension and remove '_test' suffix
        original_image_name = os.path.splitext(os.path.basename(original_image_path))[0]
        original_image_name = original_image_name.replace('_test', '')

        # Load the texture image
        texture_image = cv2.imread(original_image_path)
        if texture_image is None:
            logging.error(f"Error loading image {original_image_path}")
            return

        texture_image_rgb = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)  # Convert for plt

        # Convert the image to HSV color space for better color segmentation
        hsv_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2HSV)

        # Load the mask
        mask_file = os.path.join(input_dir, f'{original_image_name}_mask.png')
        if not os.path.exists(mask_file):
            logging.error(f"Mask file not found: {mask_file}")
            return

        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logging.error(f"Error loading mask {mask_file}")
            return

        # Load the mask
        mask_file_mid = os.path.join(input_dir, f'{original_image_name}_mask_mid.png')
        if not os.path.exists(mask_file_mid):
            logging.error(f"Mask file not found: {mask_file_mid}")
            return

        mask_mid = cv2.imread(mask_file_mid, cv2.IMREAD_GRAYSCALE)
        if mask_mid is None:
            logging.error(f"Error loading mask {mask_file_mid}")
            return

        # Define the color range for the black area (adjust these values if needed)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])

        # Create masks for the gray area and black area
        black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

        # Create a mask for orange areas excluding black areas and widened gray areas
        orange_mask = cv2.bitwise_and(cv2.bitwise_not(black_mask), cv2.bitwise_not(mask))

        orange_mask_mid = cv2.bitwise_not(cv2.bitwise_and(mask_mid, cv2.bitwise_not(mask))) #mid_mask 
        # Create a kernel for erosion
        kernel = np.ones((15, 15), np.uint8)  # Kernel size can be adjusted (e.g., 3x3, 7x7)

        # Compute the mid_mask (as in your original code)
        orange_mask_mid = cv2.morphologyEx(orange_mask_mid, cv2.MORPH_CLOSE, kernel)
        #orange_mid_mask = cv2.erode(orange_mask_mid, kernel, iterations=1)

        #orange_mask_mid = cv2.bitwise_not(cv2.bitwise_not(cv2.bitwise_not(mask)))
        
        # Invert the mask so that black regions become white and vice versa
        inverted_mask = cv2.bitwise_not(orange_mask_mid)

        # Create a white image with the same size as the original image
        white_image = np.ones_like(texture_image) * 255

        # Use the inverted mask to keep the pixels in the original image where the mask is black
        result = cv2.bitwise_and(texture_image, texture_image, mask=inverted_mask)

        # Use the mask to replace the other pixels with white
        result = cv2.add(result, cv2.bitwise_and(white_image, white_image, mask=orange_mask_mid))

        # Filter orange mask by size
        min_orange_area = 1000  # Minimum area in pixels to keep
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(orange_mask, connectivity=8)
        filtered_orange_mask = np.zeros_like(orange_mask)
        filtered_out_mask = np.zeros_like(orange_mask)  # Mask for filtered-out areas

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_orange_area:
                filtered_orange_mask[labels == i] = 255
            else:
                filtered_out_mask[labels == i] = 255

        # Combine gray and orange masks
        combined_mask = cv2.bitwise_or(mask, filtered_orange_mask)
        combined_mask2 = cv2.bitwise_or(combined_mask, black_mask)

        # Find the largest orange area in the filtered orange mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered_orange_mask, connectivity=8)

        # Identify the largest area (excluding the background)
        largest_orange_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        # Create a mask for the largest orange area
        largest_orange_mask = (labels == largest_orange_idx).astype(np.uint8) * 255

        # Extract the largest orange area from the original image
        largest_orange_area = cv2.bitwise_and(texture_image, texture_image, mask=largest_orange_mask)

        # Find a valid 100x100 pixel patch from the largest orange area
        orange_patch = find_valid_patch(largest_orange_area, largest_orange_mask, patch_size=100)
        orange_patch = cv2.resize(orange_patch, (300, 300))

        # Get the coordinates of the widened gray area
        gray_coords = np.column_stack(np.where(mask == 255))

        # Inpaint the gray area using the larger patch
        result_image = texture_image.copy()
        patch_height, patch_width, _ = orange_patch.shape

        for coord in gray_coords:
            i, j = coord
            if (i < result_image.shape[0]) and (j < result_image.shape[1]):
                result_image[i, j] = orange_patch[i % patch_height, j % patch_width]

        # Apply kernel to the result image
        gradient_result_image = apply_kernel(result_image, kernel_size=10, mask=combined_mask)

        # Combine filtered-out orange mask with black mask and widened gray mask
        filter_orange_gray_black = cv2.bitwise_or(filtered_out_mask, black_mask)
        filter_orange_gray_black = cv2.bitwise_or(filter_orange_gray_black, mask)

        # Save and plot the results
        images_to_save = {
            '': texture_image,
            'result': result,
            'black_mask': black_mask,
            'gray_mask': mask,
            'orange_mask': orange_mask,
            'orange_mask_mid': orange_mask_mid,
            'filtered_orange_mask': filtered_orange_mask,
            'filtered_out_mask': filtered_out_mask,
            'gradient_result_image': gradient_result_image,
            'filter_orange_gray_black': filter_orange_gray_black
        }

        # Save and plot images
        save_and_plot_images(images_to_save, output_dir, original_image_name)

        logging.info(f"Processed image: {original_image_name}")
        print(f"Processed image: {original_image_name}")

        # Clear memory
        del texture_image, texture_image_rgb, hsv_image, black_mask
        del orange_mask, filtered_orange_mask
        del orange_mask_mid
        del filtered_out_mask, combined_mask, combined_mask2, largest_orange_mask
        del largest_orange_area, orange_patch, gray_coords, result_image, gradient_result_image
        del filter_orange_gray_black, images_to_save

        gc.collect()

    #except Exception as e:
    else: 
        logging.error(f"Error processing image {image_file}: {e}")

# Function to process a batch of images
def process_batch(batch_number, image_batch):
    logging.info(f"Processing batch {batch_number}/{num_batches} with {len(image_batch)} images.")
    for image_file in image_batch:
        process_image(image_file, input_path, output_path)


print("num_batches : ", num_batches)
# Create and start the processing batches
for i in range(num_batches):
    start_index = i * batch_size
    end_index = min(start_index + batch_size, total_files)
    batch_files = image_files[start_index:end_index]
    process_batch(i + 1, batch_files)

    if i < num_batches - 1:
        logging.info(f"Batch {i + 1} complete. Waiting for {delay_between_batches} seconds before starting the next batch.")
        time.sleep(delay_between_batches)

logging.info('Batch processing completed.')
