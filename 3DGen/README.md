# Generation of 3D Meshes Dataset

## Description

This project focuses on generating and utilizing 3D meshes, particularly of fruits and vegetables. It is divided into two main stages:

1. **Generation of 3D Meshes**: Creating detailed 3D meshes of various fruits and vegetables.
2. **Simulation and Dataset Creation**: Using the generated meshes to simulate scenes with BlenderProc and create datasets, such as NOCS CAMERA.

## Prerequisites

Ensure you have the following installed:

- **Ubuntu**: The project is tested on Ubuntu.
- **Stable Diffusion 3**: For generating high-quality datasets of images.
- **Instant Meshes**: For converting images into 3D meshes.
- **Blender**: Needed for refining 3D meshes.
- **BlenderProc**: To simulate scenes and create the final datasets.

## Pipeline Overview

The project pipeline is divided into two main stages:
1. **Image Generation**: Using Stable Diffusion to generate a large dataset of images.
2. **3D Mesh Generation and Simulation**: Converting the images into 3D meshes and simulating scenes with BlenderProc.

## Detailed Steps

### Step 1: Generating a Dataset of Images using Stable Diffusion 3

1. **Create a Virtual Environment**:
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   ```

2. **Install Required Packages**:
   ```bash
   pip install diffusers torch transformers ipywidgets accelerate protobuf sentencepiece
   ```

3. **Prepare Directories**:
   - Create a folder to store the generated images and update the path in the `image_generation.py` script accordingly.

4. **Run Image Generation Script**:
   ```bash
   python image_generation.py
   ```
   In this project, 10,000 images were generated for 50 categories of fruits using the `stabilityai/stable-diffusion-3-medium-diffusers` model.

### Step 2: Generating 3D Meshes with Instant Meshes

1. **Install Instant Meshes**:
   Follow the [installation guide](https://github.com/TencentARC/InstantMesh) to install Instant Meshes.

2. **Run Instant Meshes**:
   ```bash
   python run.py configs/instant-mesh-large.yaml path_to_your_saved_images --save_video --export_texmap
   ```
   Replace `path_to_your_saved_images` with the folder where your generated images are stored. This will generate the 3D meshes along with texture maps.

3. **Important**:
   - The 3D meshes (in `.obj`, `.mtl`, and `.png` formats) will be saved in a specified folder. This folder will be crucial for the next steps, so make sure to note its path carefully or copy them into another folder you create and note the new path.

### Step 3: Installing and Using BlenderProc

1. **Clone the BlenderProc Repository**:
   ```bash
   git clone https://github.com/DLR-RM/BlenderProc
   cd BlenderProc
   ```

2. **Install BlenderProc**:
   ```bash
   pip install -e .
   ```

3. **Copy Required Scripts**:
   - Before running `select_poly_all.py`, copy `select_poly_all.py`, `update_uv_all.py`, and `update_json.py` to the `BlenderProc/blenderproc/scripts` directory.

4. **Create Output Directories**:
   - Create the necessary output directories:
     ```bash
     mkdir -p BlenderProc/blenderproc/scripts/output BlenderProc/blenderproc/scripts/output_json_updated
     ```

5. **Change Directory to BlenderProc Scripts**:
   ```bash
   cd blenderproc/scripts
   ```

6. **Run `select_poly_all.py`**:
   - Update `mesh_dir` to point to the folder with your 3D models and set `destination_folder` to your desired output folder (e.g., `/your_home_path/BlenderProc/blenderproc/scripts/output`).
   ```bash
   blenderproc debug select_poly_all.py
   ```

### Step 4: Additional Dependencies and Pre-Inpaint Processing

**Note**: Before proceeding, **cd out** of the BlenderProc directory.

1. **Install Required Packages**:
   ```bash
   pip install opencv-python numpy matplotlib
   ```

2. **Run `rename_file.py`**:
   - Ensure `folder_path` points to the directory containing the 3D meshes and mask files generated in Step 3.
   ```bash
   python rename_file.py
   ```

3. **Run `run.py`**:
   - Make sure that `input_path` and `output_path` match the `folder_path` set in `rename_file.py`.
   ```bash
   python run.py
   ```

### Step 5: Setting Up ZITS Inpainting

1. **Navigate to the ZITS Inpainting GitHub Page**:
   - Go to the [ZITS Inpainting GitHub repository](https://github.com/DQiaole/ZITS_inpainting) and follow their installation instructions.

2. **Prepare Directories**:
   - If the `images`, `masks`, and `output` folders do not exist inside the ZITS directory, create them:
     ```bash
     mkdir -p ZITS_inpainting/images ZITS_inpainting/masks ZITS_inpainting/output
     ```

3. **Copy Files to ZITS Directory**:
   - Use `copy_to_zits.py` to copy the 3D meshes and masks into the appropriate folders inside `ZITS_inpainting`:
     ```bash
     python copy_to_zits.py
     ```
   - Set the following paths in `copy_to_zits.py`:
     - `source_dir`: Directory where your 3D meshes and mask files are located.
     - `dest_dir_black`: Set to `ZITS_inpainting/masks`.
     - `dest_dir_apple`: Set to `ZITS_inpainting/images`.

4. **Run `single_image_test_script.py`**:
   - After setting up ZITS and copying the files, run the script:
     ```bash
     python single_image_test_script.py
     ```
   - The inpainted textures will be saved in the `output` folder inside the `ZITS_inpainting` directory.

5. **Troubleshooting**:
   - If you encounter installation issues with ZITS, refer to their [GitHub page](https://github.com/DQiaole/ZITS_inpainting) for assistance.

### Step 6: Final Processing with Replace Script

1. **Run `replace.py`**:
   - Set the following paths:
     - `input_dir_original`: Directory where your 3D meshes and mask files are located.
     - `input_dir_mask`: The same directory as `input_dir_original`.
     - `input_dir_inpainted`: Path to the `output` folder from ZITS_inpainting.
     - `output_dir`: Set this to `/your_home_directory/BlenderProc/blenderproc/scripts/output`.
   ```bash
   blenderproc debug replace.py
   ```

### Steps 7 & 8: Final UV and JSON Updates

1. **Navigate to BlenderProc Scripts**:
   ```bash
   cd BlenderProc/blenderproc/scripts
   ```

2. **Run `update_json.py`**:
   - Set the paths:
     - `input_folder = '/your_home_directory/BlenderProc/blenderproc/scripts/output'`
     - `output_folder = '/your_home_directory/BlenderProc/blenderproc/scripts/output_json_updated'`
   ```bash
   blenderproc debug update_json.py
   ```

3. **Run `update_uv_all.py`**:
   - Set the following paths:
     - `mesh_dir = '/your_home_directory/BlenderProc/blenderproc/scripts/output'`
     - `destination_folder = '/your_home_directory/BlenderProc/blenderproc/scripts/output_json_updated'`
     - `json_directory = '/your_home_directory/BlenderProc/blenderproc/scripts/output_json_updated'`
   ```bash
   blenderproc debug update_uv_all.py
   ```

Certainly! Here’s the revised Step 2 for the Simulation and Dataset Creation section:

---

## Simulation and Dataset Creation

### Step 1: Setting Up BlenderProc Renderer

1. **Replace Renderer Files**:
   - In shapo (fruit edition repo) Replace the existing files in the `BlenderProc/blenderproc/renderer` directory with the files from your `renderer` folder. This ensures you have the updated configuration for fruit simulation.



### Step 2: Update Simulation Scripts

1. **Copy New Scripts**:
   - Place the new scripts, `create_images_without_background.py` and `create_images_with_background.py`, into the `shapo (fruit edition repo)` 


2. **Modify `create_scene.py`**:
   - Update `create_scene.py` to use the new scripts. Open `create_scene.py` and ensure it references `create_images_without_background.py` and `create_images_with_background.py` as needed. This might involve modifying import statements or function calls to integrate the new scripts.



