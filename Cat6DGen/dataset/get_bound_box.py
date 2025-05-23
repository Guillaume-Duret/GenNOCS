import blenderproc as bproc
import argparse
import os
import glob
import numpy as np
import math
import bpy


def process_mesh(obj_path):
    # Load and orient the mesh
    scene_objs = bproc.loader.load_obj(obj_path)
    mesh_obj = scene_objs[0]
    mesh_obj.set_rotation_euler([math.radians(0), 0, 0])
    bpy.context.view_layer.objects.active = mesh_obj.blender_obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    # Compute bounding box
    bbox = mesh_obj.get_bound_box()
    min_coords = np.min(bbox, axis=0)
    max_coords = np.max(bbox, axis=0)
    return min_coords, max_coords


def main(input_dir):
    bproc.init()
    # Iterate through subfolders
    for entry in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, entry)
        if not os.path.isdir(folder_path):
            continue
        obj_files = glob.glob(os.path.join(folder_path, '3d', '*.obj'))
        if not obj_files:
            print(f"No .obj file found in {entry}")
            continue
        for obj_file in obj_files:
            min_c, max_c = process_mesh(obj_file)
            print(f"  Min coords: {min_c}")
            print(f"  Max coords: {max_c} \n \n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch extract bounding boxes from OBJ meshes in subfolders.')
    parser.add_argument('input_dir', help='Path to the parent directory with .obj subfolders')
    args = parser.parse_args()
    main(args.input_dir)