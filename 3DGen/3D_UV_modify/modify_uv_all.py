import blenderproc as bproc
import numpy as np
import bpy
import bmesh
import json
from mathutils import Vector
from PIL import Image, ImageDraw
import os
import shutil
import fnmatch

# Initialize BlenderProc
bproc.init()

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
parser.add_argument(
    "--json_path",
    type=str,
    required=True,
    help="Path to the directory containing the input"
)

# Parse the arguments
args = parser.parse_args()

# Path to the directory containing your mesh files
mesh_dir = args.input_path #"output_test_NOCS_data/"

# Destination folder for the exported files
destination_folder = args.output_path # "output_test_NOCS_json_updated"
json_directory = args.json_path # "output_test_NOCS_json_updated"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)
camera_pose_added = False

# Iterate through the directory to find all .obj files
print("os.listdir(mesh_dir) : ", os.listdir(mesh_dir))
for file_name in os.listdir(mesh_dir):
    if file_name.endswith(".obj"):
        print("file_name : ", file_name)

for file_name in os.listdir(mesh_dir):
    if file_name.endswith(".obj"):
        # Construct file paths for the .obj, .mtl, and .png files
            # Path to your mesh files
            obj_file = os.path.join(mesh_dir, file_name)
            base_name = os.path.splitext(file_name)[0]
            mtl_file = os.path.join(mesh_dir, base_name + ".mtl")
            png_file = os.path.join(mesh_dir, base_name + ".png")

            # Load the mesh object
            objs = bproc.loader.load_obj(obj_file)

            # Apply the material to the object
            for obj in objs:
                if obj.get_name().endswith(".mtl"):
                    obj.set_cp("use_nodes", True)
                    mat = obj.get_materials()[0]
                    mat.new_texture("Diffuse", png_file)

            # Create a point light next to the object
            light = bproc.types.Light()
            light.set_location([0, 4, 0])
            light.set_energy(300)

            # Set the camera to be in front of the object
            # Add the camera pose only if it hasn't been added yet
            if not camera_pose_added:
                cam_pose = bproc.math.build_transformation_mat([0, 5, 0], [-np.pi / 2, -np.pi, 0])
                bproc.camera.add_camera_pose(cam_pose)
                camera_pose_added = True 

            # Render the scene
            data = bproc.renderer.render()
            updated_uvs = {}
            for filename in os.listdir(json_directory):
                pattern = f"updated_{base_name}_selected_polys_uv.json"
                if fnmatch.fnmatch(filename, pattern):
                    file_path = os.path.join(json_directory, filename)
                    with open(file_path, "r") as f:
                        try:
                            updated_uvs = json.load(f)
                            print(f"Loaded JSON from {filename}:")
                            # print(updated_uvs)
                        except json.JSONDecodeError as e:
                            print(f"Error reading {filename}: {e}")

            # Switch to bpy to access edit mode and polygons
            for obj in objs:
                bpy_object = bpy.data.objects[obj.get_name()]  # Link BlenderProc object to bpy object
                bpy.context.view_layer.objects.active = bpy_object

                # Ensure object is in 'OBJECT' mode to access mesh data
                bpy.ops.object.mode_set(mode='OBJECT')

                # Get mesh data
                mesh = bpy_object.data
                camera = bpy.data.objects['Camera']

                # Ensure there is an active UV layer
                if not mesh.uv_layers:
                    print(f"Object {bpy_object.name} has no UV layers.")
                    continue

                uv_layer = mesh.uv_layers.active.data

                # Switch to 'EDIT' mode to use bmesh
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.reveal()  # Ensure all faces are selectable
                bpy.ops.mesh.select_all(action='DESELECT')

                # Create a BMesh object to work with
                bm = bmesh.from_edit_mesh(mesh)
                bm.faces.ensure_lookup_table()

                # Create a dictionary to hold selected polygons UV coordinates
                selected_polys_uv = {}
                selected_poly = []

                # Create a blank mask image
                width, height = 1024, 1024  # Size of the output mask image
                mask_image = Image.new('L', (width, height), 0)  # 'L' mode is for grayscale images
                draw = ImageDraw.Draw(mask_image)

                # Step 1: Find the minimum and maximum Z-values for all faces relative to the camera
                z_values = []
                for face in bm.faces:
                    # Calculate the world coordinate of the face center
                    face_center_world = bpy_object.matrix_world @ face.calc_center_median()
                    # Transform world coordinate to camera coordinate
                    face_center_camera = camera.matrix_world.inverted() @ face_center_world
                    # Append the Z-coordinate in camera space to the list
                    z_values.append(face_center_camera.z)

                min_z, max_z = min(z_values), max(z_values)

                # Define the threshold to select faces that are 10% lower than the highest Z-value
                threshold_z = max_z - (max_z - min_z) * 0.15

                # Step 2: Select faces that meet the threshold condition
                for face, z in zip(bm.faces, z_values):
                    # If the face's Z-coordinate is within the threshold, select it
                    if z > threshold_z:
                        face.select = True
                        selected_poly.append(face.index)

                # Check if any faces were selected
                if not selected_poly:
                    print("No faces meet the selection criteria.")
                else:
                    print(f"{len(selected_poly)} faces selected.")


                bmesh.update_edit_mesh(mesh)

                # Switch back to 'OBJECT' mode to access the final data
                bpy.ops.object.mode_set(mode='OBJECT')

                mesh = bpy_object.data
                if not mesh.uv_layers.active:
                    print("No UV layers found.")
                uv_layer = mesh.uv_layers.active.data

              
                # Apply updated UV coordinates from the JSON file
                for poly in mesh.polygons:
                    if str(poly.index) in updated_uvs:
                        updated_coords = updated_uvs[str(poly.index)]
                        for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                            uv = uv_layer[loop_index].uv
                            # Convert JSON UVs to Blender UVs
                            uv.x = updated_coords[loop_index - poly.loop_start][0] / (width - 1)
                            uv.y = 1 - updated_coords[loop_index - poly.loop_start][1] / (height - 1)
                            uv_layer[loop_index].uv = uv

            
            # Export only the modified object to OBJ format
            bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
            for obj in objs:
                bpy.data.objects[obj.get_name()].select_set(True)  # Select only the mesh objects

            output_obj_file = os.path.join(destination_folder, f"{base_name}.obj")
            bpy.ops.export_scene.obj(filepath=output_obj_file, use_selection=True, use_materials=True)

            # The MTL file is automatically saved alongside the OBJ file
            output_mtl_file = output_obj_file.replace(".obj", ".mtl")

            # Copy the original PNG file to the destination folder
            output_png_file = os.path.join(destination_folder, os.path.basename(png_file))
            shutil.copy(png_file, output_png_file)

            # Modify the MTL file to refer to the PNG in the destination folder
            with open(output_mtl_file, 'r') as file:
                mtl_data = file.readlines()

            # Replace the path of the texture
            for i, line in enumerate(mtl_data):
                if line.startswith('map_Kd'):
                    mtl_data[i] = f"map_Kd {base_name}.png\n"
                    break

            # Write the changes back to the MTL file
            with open(output_mtl_file, 'w') as file:
                file.writelines(mtl_data)

            print(f"Exported OBJ file: {output_obj_file}")
            print(f"Exported MTL file: {output_mtl_file}")
            print(f"Copied PNG file to: {output_png_file}")

print("Processing complete Done !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.")
