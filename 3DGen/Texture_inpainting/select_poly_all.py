import blenderproc as bproc
import numpy as np
import bpy
import bmesh
import json
import shutil
import os
from PIL import Image, ImageDraw
import fnmatch
import argparse

# Path to the directory containing your mesh files

# Create the parser
parser = argparse.ArgumentParser(description="Process mesh directory and output folder.")

# Add arguments
parser.add_argument(
    "--mesh_dir",
    type=str,
    required=True,
    help="Path to the directory containing your mesh files"
)
parser.add_argument(
    "--destination_folder",
    type=str,
    required=True,
    help="Destination folder for the exported files"
)

# Parse the arguments
args = parser.parse_args()

mesh_dir = args.mesh_dir #"/lustre/fsn1/projects/rech/tya/ubn15wo/3D_GEN/test_NOCS_data"
# Destination folder for the exported files
destination_folder = args.destination_folder #"/lustre/fsn1/projects/rech/tya/ubn15wo/3D_GEN/output_test_NOCS_data"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Initialize BlenderProc
bproc.init()
camera_pose_added = False  # bool mark that a camera pose has been added

# Iterate through the directory to find all .obj files
for file_name in os.listdir(mesh_dir):
    if file_name.endswith(".obj"):
        # Construct file paths for the .obj, .mtl, and .png files
            # Path to your mesh files

            print("------------------------")
            print("IN FOR LOOP 0")
            print("------------------------")
            #for file_name in os.listdir(mesh_dir)
            obj_file = os.path.join(mesh_dir, file_name)
            base_name = os.path.splitext(file_name)[0]
            mtl_file = os.path.join(mesh_dir, base_name + ".mtl")
            png_file = os.path.join(mesh_dir, base_name + ".png")

            # Extract filename without extension for use in saving the OBJ file
            base_name = os.path.splitext(os.path.basename(obj_file))[0]

            # Load the mesh object
            objs = bproc.loader.load_obj(obj_file)
            
            print("objs : ", objs) 
            # Apply the material to the object
            for obj in objs:
                if obj.get_name().endswith(".mtl"):
                    print("------------1-------------in first for : obj : ", obj)
                    obj.set_cp("use_nodes", True)
                    mat = obj.get_materials()[0]
                    mat.new_texture("Diffuse", png_file)

            # Create a point light next to the object
            light = bproc.types.Light()
            light.set_location([0, 4, 0])
            light.set_energy(300)

            # Check if there are any existing camera poses
            if not camera_pose_added :  # Check if no camera poses exist
                # Set the camera pose (only if no poses exist)
                cam_pose = bproc.math.build_transformation_mat([0, 5, 0], [-np.pi / 2, -np.pi, 0])
                bproc.camera.add_camera_pose(cam_pose)  # Add a single pose for this object
                camera_pose_added = True  # Mark that a camera pose has been added

            print("before render")
            # Render the scene
            data = bproc.renderer.render()
            print("after render")

            # Switch to bpy to access edit mode and polygons
            for obj in objs:

                print("-----2-------in second for : obj : ", obj)
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
                selected_polys_uv_mid = {}
                selected_poly = []
                selected_poly_mid = []

                # Create a blank mask image (not used in the final script)
                width, height = 1024, 1024  # Size of the output mask image
                mask_image = Image.new('L', (width, height), 0)  # 'L' mode is for grayscale images
                draw = ImageDraw.Draw(mask_image)

                mask_image_mid = Image.new('L', (width, height), 0)  # 'L' mode is for grayscale images
                draw_mid = ImageDraw.Draw(mask_image_mid)


                """
                while selected_poly == [] : 
                    # Iterate over BMesh faces
                    count = 0
                    for face in bm.faces:
                        # Compute face center in world coordinates
                        face_center_world = bpy_object.matrix_world @ face.calc_center_median()
                        # Transform world coordinates to camera coordinates
                        face_center_camera = camera.matrix_world.inverted() @ face_center_world

                        # Check if face is in front of the camera
                        if face_center_camera.z > -5 + count*2:
                            # Select the face
                            face.select = True
                            selected_poly.append(face.index)
                """

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
                threshold_z_mid = max_z - (max_z - min_z) * 0.5 #take only 50% to reduce risk of too musch variation of color

                # Step 2: Select faces that meet the threshold condition
                for face, z in zip(bm.faces, z_values):
                    # If the face's Z-coordinate is within the threshold, select it
                    if z > threshold_z:
                        face.select = True
                        selected_poly.append(face.index)
                    if z > threshold_z_mid:
                        face.select = True
                        selected_poly_mid.append(face.index)

                # Check if any faces were selected
                if not selected_poly:
                    print("No faces meet the selection criteria.")
                else:
                    print(f"{len(selected_poly)} faces selected.")

                # Check if any faces were selected
                if not selected_poly_mid:
                    print("No faces meet the selection criteria.")
                else:
                    print(f"{len(selected_poly_mid)} mid faces selected.")

                bmesh.update_edit_mesh(mesh)

                # Switch back to 'OBJECT' mode to access the final data
                bpy.ops.object.mode_set(mode='OBJECT')

                mesh = bpy_object.data
                if not mesh.uv_layers.active:
                    print("No UV layers found.")
                uv_layer = mesh.uv_layers.active.data

                for poly in mesh.polygons:
                    if poly.index in selected_poly:
                        # Collect UV coordinates for the selected polygon
                        poly_uvs = []
                        for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                            uv = uv_layer[loop_index].uv
                            poly_uvs.append((int(uv.x * (width - 1)), int((1 - uv.y) * (height - 1))))  # Note: Flip Y-axis

                        # Debug print UV coordinates
                        #print(f"Polygon {poly.index} UVs: {poly_uvs}")

                        # Draw the polygon on the mask (not used in the final script)
                        if poly_uvs:  # Ensure there are UV coordinates to draw
                            draw.polygon(poly_uvs, outline=255, fill=255)
                        selected_polys_uv[poly.index] = poly_uvs
                    if poly.index in selected_poly_mid:
                        # Collect UV coordinates for the selected polygon
                        poly_uvs = []
                        for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                            uv = uv_layer[loop_index].uv
                            poly_uvs.append((int(uv.x * (width - 1)), int((1 - uv.y) * (height - 1))))  # Note: Flip Y-axis

                        # Debug print UV coordinates
                        #print(f"Polygon {poly.index} UVs: {poly_uvs}")

                        # Draw the polygon on the mask (not used in the final script)
                        if poly_uvs:  # Ensure there are UV coordinates to draw
                            draw_mid.polygon(poly_uvs, outline=255, fill=255)
                        selected_polys_uv_mid[poly.index] = poly_uvs

                # Save the updated UV coordinates of selected polygons to a file
                with open(os.path.join(destination_folder, f"{base_name}_selected_polys_uv.json"), "w") as f:
                    json.dump({int(k): v for k, v in selected_polys_uv.items()}, f, indent=4)

                # Save the updated UV coordinates of selected polygons to a file
                with open(os.path.join(destination_folder, f"{base_name}_selected_polys_uv_mid.json"), "w") as f:
                    json.dump({int(k): v for k, v in selected_polys_uv_mid.items()}, f, indent=4)

                mask_image.save(os.path.join(destination_folder, f"{base_name}_mask.png"))
                print(f"-----3------save {base_name}_mask.png")

                mask_image_mid.save(os.path.join(destination_folder, f"{base_name}_mask_mid.png"))
                print(f"-----3------save {base_name}_mask_mid.png")


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

print("Done with the process !!!!!!!!!!!!!.")
