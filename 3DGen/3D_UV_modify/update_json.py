import blenderproc as bproc
import json
import math
import copy
import os
import fnmatch
import random
import numpy as np
import bpy 
import bmesh
from scipy.spatial import KDTree
from collections import deque
import argparse

def calculate_centroid(polygon):
    num_vertices = len(polygon)
    if num_vertices == 0:
        return (0, 0)
    
    sum_u = sum(v[0] for v in polygon)
    sum_v = sum(v[1] for v in polygon)
    
    centroid_u = sum_u / num_vertices
    centroid_v = sum_v / num_vertices
    
    return (centroid_u, centroid_v)

def read_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def calculate_euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def find_neighbors_by_id_difference(data, target_id, id_threshold=5, exclude_ids=None):
    if exclude_ids is None:
        exclude_ids = set()
    
    target_id_int = int(target_id)
    neighbors = []
    
    for key in data.keys():
        key_int = int(key)
        if key != target_id and key not in exclude_ids and 0 < abs(target_id_int - key_int) < id_threshold:
            neighbors.append(key)
    
    return neighbors


def calculate_average_centroid(neighbors, centroids):
    if not neighbors:
        return (0, 0)
    
    sum_u = sum(centroids[str(neighbor)][0] for neighbor in neighbors)
    sum_v = sum(centroids[str(neighbor)][1] for neighbor in neighbors)
    
    avg_u = sum_u / len(neighbors)
    avg_v = sum_v / len(neighbors)
    
    return (avg_u, avg_v)


# Step 3: Define function to get neighboring faces within a radius using KD-Tree
def get_neighbors_within_radius_kdtree(face_id, radius,centroids,kd_tree,face_index_map,face_ids):
    # Get the centroid of the target face
    target_centroid = centroids[face_id]
    
    # Query the KD-Tree for all centroids within the radius
    indices = kd_tree.query_ball_point(target_centroid, radius)
    
    # Map centroid indices back to face indices
    neighboring_faces = [face_index_map[idx] for idx in indices if idx != face_id]
    #print the type of neighboring_faces
    # print(type(neighboring_faces[0]))
    # print(type(face_ids[0]))
    # Filter out neighbors not in the face_ids list
    valid_neighbors = [face for face in neighboring_faces if str(face) in face_ids]
    
    return valid_neighbors

def process_file(filename, output_filename,mesh,radius):
    # Read JSON data
    data = read_json_file(filename)
    face_ids= list(data.keys())
    updated_data = copy.deepcopy(data)
    updated_polygons = set()
    ####################################################################
    # Step 1: Calculate centroids for all faces
    centroids = []
        # Calculate centroids
    centroids_uv = {key: calculate_centroid(polygon) for key, polygon in data.items()}
    face_index_map = {}  # To map face indices to centroid indices

    for face in mesh.polygons:
        centroid = np.mean([mesh.vertices[vertex].co for vertex in face.vertices], axis=0)
        centroids.append(centroid)
        face_index_map[len(centroids) - 1] = face.index

    centroids = np.array(centroids)

    # Step 2: Build KD-Tree using centroids
    kd_tree = KDTree(centroids)
    
    updated_polygons = set()  # Skip polygons that have already been updated
    # neighbors = []
    i=0
    for face_id in face_ids:
         #print Processing face number i out of lenght of data.keys
        i+=1
        #print(f"Processing of id {face_id}  face number {i} out out of {len(face_ids)}")

        if face_id in updated_polygons:
            #print(f"Skipping {face_id}, already updated.")
            continue

        if int(face_id) in face_index_map.values():
            centroid_idx = list(face_index_map.values()).index(int(face_id))
            neighbors = get_neighbors_within_radius_kdtree(centroid_idx, radius, centroids, kd_tree, face_index_map,face_ids)
            #print("Neighboring face indices within radius:", neighbors)

        if not neighbors:
            continue

        nei = []
        neighbors_taille = len(neighbors)
        
        # Filter out far-away neighbors
        for neighbor1 in neighbors:
            cmpt = 0
            for neighbor2 in neighbors:
                distance_nei = calculate_euclidean_distance(centroids_uv[str(neighbor1)], centroids_uv[str(neighbor2)])
                if distance_nei > 100:  # Threshold for distance
                    cmpt += 1
            
            if cmpt >= neighbors_taille / 2:
                neighbors.remove(neighbor1)
                nei.append(str(neighbor1))


        for neighbor1 in nei:
            # neighbors.remove(neighbor1)
            updated_data[neighbor1] = data[str(neighbors[0])]
            updated_polygons.add(neighbor1)


        # print("Neighboring face indices within radius:", neighbors)
        if face_id in neighbors:
            neighbors.remove(face_id)
       
        average_centroid = calculate_average_centroid(neighbors, centroids_uv)
        target_centroid = centroids_uv[face_id]
        distance = calculate_euclidean_distance(target_centroid, average_centroid)
        
        if distance > 100:
            random_neighbor = neighbors[0]
            # Update the polygon's coordinates to the chosen neighbor's coordinates
            updated_data[face_id] = data[str(random_neighbor)]
            updated_polygons.add(face_id)  # Mark this polygon as updated
    
    # Save the updated data to a new JSON file
    with open(output_filename, 'w') as file:
        json.dump(updated_data, file, indent=4)
    
    print(f"Updated polygon data saved to {output_filename}")

def main():

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

    input_folder = args.input_path #"/lustre/fsn1/projects/rech/tya/ubn15wo/3D_GEN/test_NOCS_data"
    # Destination folder for the exported files
    output_folder = args.output_path 

    #input_folder = '/lustre/fsn1/projects/rech/tya/ubn15wo/3D_GEN/output_test_NOCS_data'
    #output_folder = '/lustre/fsn1/projects/rech/tya/ubn15wo/3D_GEN/output_test_NOCS_json_updated'
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    

    # Iterate through all JSON files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('uv.json'):
            pattern = filename
        
            # Extracting the base name (first word and number)
            base_name = pattern.split('_')[0] + '_' + pattern.split('_')[1] + '_' + pattern.split('_')[2]
            pattern1 = base_name + ".obj"
            print("filename : ", filename)
            print("base_name : ", base_name)
            print("pattern : ", pattern)
            print("pattern split : ", pattern.split('_'))

            # Construct the full path for the output file
            output_filepath = os.path.join(output_folder, f'updated_{filename}')

            # Check if the output file already exists
            if os.path.exists(output_filepath):
                print(f"Output file {output_filepath} already exists. Skipping...")
                continue  # Skip to the next file

            # If the output file does not exist, proceed with processing
            if fnmatch.fnmatch(filename, pattern):
                # Specify the path to your file
                file_path = os.path.join(input_folder, pattern1)

                objs = bproc.loader.load_obj(file_path)

                for obj in objs:
                    bpy_object = bpy.data.objects[obj.get_name()]  # Link BlenderProc object to bpy object
                mesh = bpy_object.data

                radius = 0.04  # Set the desired radius
                input_filepath = os.path.join(input_folder, filename)

                # Call your processing function
                process_file(input_filepath, output_filepath, mesh, radius)


if __name__ == "__main__":
    main()

print("Script completed successfully.")

