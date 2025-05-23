import blenderproc as bproc
import argparse
import numpy as np
import cv2
import os
import sys
import json
import random
from blenderproc.python.loader.ReplicaLoader import load_replica
from blenderproc.python.loader.HavenEnvironmentLoader import set_world_background_hdr_img, get_random_world_background_hdr_img_path_from_haven
from mathutils import Vector, Euler, Matrix
from PIL import Image
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default="/home/mazurrrak/ECL/Semester2/PO/Omni6D/output", help="Path to where the final files, will be saved, could be examples/basics/basic/output")
parser.add_argument('--split', default="train", help="dataset split you want to generate, could be train, val, test, test_unseen")
parser.add_argument('--scene', default="<path-to-replica>", help="Path to where the final files, will be saved, could be examples/basics/basic/output")
parser.add_argument('--objects', default="/home/mazurrrak/ECL/Semester2/PO/Omni6D/data/obj_models_small/", help="Path to where the final files, will be saved, could be ")

args = parser.parse_args()
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
split = args.split

if split == 'train':
    output_split = split
    max_file = 100000
elif split == 'val':
    output_split = split
    max_file = 2000
elif split == 'test':
    output_split = split
    max_file = 2000
elif split == 'test_unseen':
    output_split = split
    max_file = 2000

os.makedirs(os.path.join(args.output_dir, output_split), exist_ok=True)
files = os.listdir(os.path.join(args.output_dir, output_split))
width = 5
whole_files = [str(id).zfill(width) for id in range(max_file)]
rest_files = list(set(whole_files) - set(files))
file_id = random.choice(rest_files)
print("file_id:", file_id)
output_dir = os.path.join(args.output_dir, output_split, file_id)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    print(f"Directory '{output_dir}' already exists. Goodbye the program.")
    sys.exit()

pic_num = 10
roomscale = 10
room_dict = {'office_2': [[-8, -7, -8], [10, 30, 4]], 
            'office_3': [[-7, -20, 0], [11, 17, 5]],
            'office_4':  [[0, -15.0, 0], [35, 15, 12]],
            'room_0': [[0.0, 0.0, 0.0], [55, 30, 5]],
            'room_1': [[-30, -5, -8], [0, 10, 2.5]],
            'room_2': [[10, -12, -12], [40, 6, -7]],
            'hotel_0': [[-25, 3, 0.0], [40, 12, 5]],
            'frl_apartment_0': [[0, -50, -6.25], [40, -30, 4]],
            }

if split == 'test_unseen':
    catId_to_name = {1: 'bottle', 2: 'bowl', 3: 'camera', 4: 'can', 5: 'laptop', 6: 'mug'}
    #catId_to_name = {1: 'bottle', 2: 'bowl', 3: 'camera', 4: 'can', 5: 'laptop2', 6: 'mug', 7: 'apples', 8: 'apricots', 9: 'bananas', 10: 'dragon', 11: 'figs', 12: 'oranges', 13: 'peaches', 14: 'pears', 15: 'pineapples', 16: 'watermelons', 17: 'artichokes', 18: 'broccoli', 19: 'cauliflower', 20: 'corn', 21: 'jalapeño', 22: 'lettuce', 23: 'mushrooms', 24: 'potatoes', 25: 'pumpkin', 26: 'tomatoes'}
else:
    catId_to_name = {1: 'bottle', 2: 'bowl', 3: 'camera', 4: 'can', 5: 'laptop', 6: 'mug'}
    #catId_to_name = {1: 'bottle', 2: 'bowl', 3: 'camera', 4: 'can', 5: 'laptop2', 6: 'mug', 7: 'apples', 8: 'apricots', 9: 'bananas', 10: 'dragon', 11: 'figs', 12: 'oranges', 13: 'peaches', 14: 'pears', 15: 'pineapples', 16: 'watermelons', 17: 'artichokes', 18: 'broccoli', 19: 'cauliflower', 20: 'corn', 21: 'jalapeño', 22: 'lettuce', 23: 'mushrooms', 24: 'potatoes', 25: 'pumpkin', 26: 'tomatoes'}

if split == 'train':
    room_list = ['office_2', 'room_0', 'hotel_0', 'frl_apartment_0']
elif split == 'val':
    room_list = ['office_4', 'room_2']
else:
    room_list = ['office_3', 'room_1']
name_to_catId = {value:key for key,value in catId_to_name.items()}

room_name = random.choice(room_list)
print("room_name:", room_name)

def create(room_name):  
    # global location_list
    bproc.init() 
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)
    room = load_replica(args.scene, room_name, use_smooth_shading = True)[0]
    size = room.get_bound_box()
    size_x = max(abs(size[:, 0]))
    size_y = max(abs(size[:, 1]))
    size_z = max(abs(size[:, 2]))
    print(size_x, size_y, size_z)
    room.set_scale([roomscale, roomscale, roomscale])
    
    model_objects = args.objects
    model_num = random.choice(list(range(4,8)))
    models = []
    cates = os.listdir(os.path.join(model_objects, split))
    print(cates)
    for cate in sorted(cates):
        cate_path = os.path.join(model_objects, split, cate)
        objects = os.listdir(cate_path)
        for obj in objects:
            obj_folder = os.path.join(cate_path, obj)
            for file in os.listdir(obj_folder):
                if file.endswith(".obj"):
                    models.append(os.path.join(obj_folder, file))
                    break
    print("Total models found:", len(models))
    objs_path = random.choices(models, k=model_num)
    print("Selected object paths:", objs_path)

    objs = []
    for obj_id in range(len(objs_path)):
        obj_path = objs_path[obj_id]
        scale_path = obj_path.replace('.obj', '.json')
        with open(scale_path, 'r') as f:
            scale_data = json.load(f) 
        obj_scale = scale_data["scale_factor"]*1000 # for scale to meter, real size + roomscale?
        print("obj_scale:", obj_scale)
        obj_name = os.path.basename(obj_path).lower()
        print("obj_name : ", obj_name)
        print(obj_path, obj_scale)
        obj = bproc.loader.load_obj(obj_path)[0]
        category_name = obj_path.split(os.sep)[-3]
        if category_name.endswith("_meshes"):
            category_name = category_name.replace("_meshes", "")
        cate_id = name_to_catId[category_name]
        obj.set_cp("category_id", cate_id)
        obj.set_cp("instance_id", obj_id + 1)
        obj.set_cp("obj_scale", obj_scale)
        objs.append(obj)

    '''点光源的设置 - Встановлення точкових джерел світла :)'''
    lights = []
    num_lights = 5
    light_class = 'POINT'
    for i in range(num_lights):
        light = bproc.types.Light()
        light.set_energy(random.uniform(500, 700)*roomscale/num_lights)
        if room_name == 'room_0':
            light.set_energy(random.uniform(550, 650)*roomscale/num_lights)
        if room_name == 'room_1':
            light.set_energy(random.uniform(400, 500)*roomscale/num_lights)
        if room_name == 'hotel_0':
            light.set_energy(random.uniform(1100, 1300)*roomscale/num_lights)
        if room_name == 'office_2':
            light.set_energy(random.uniform(1400, 1600)*roomscale/num_lights)
        if room_name == 'office_3':
            light.set_energy(random.uniform(550, 650)*roomscale/num_lights)
        light.set_type(light_class)
        center_init = np.mean(np.array([room_dict[room_name][0], room_dict[room_name][1]]), axis = 0)
        center_init[2] = room_dict[room_name][1][2]
        light_location = bproc.sampler.shell(
            center=center_init,
            radius_min=0.4 * roomscale,
            radius_max=0.6 * roomscale,
            elevation_min=0,
            elevation_max=1,
        )
        light.set_location(light_location)
        lights.append(light)

    sizes = []
    max_size = 0
    for obj in objs:
        obj_scale = obj.get_cp("obj_scale")
        size = obj.get_bound_box()
        size_x = max(abs(size[:, 0]))
        size_y = max(abs(size[:, 1]))
        size_z = max(abs(size[:, 2]))
        if obj_scale > max_size:
            max_size = obj_scale
        real_size = [size_x, size_y, size_z, obj_scale * 0.01] # 0.01 = 10* 0.001
        sizes.append(real_size)
        # 前者表示归一化模型的scale，后者表示模型从[0,1]^3单位空间中的放缩
        # obj.set_scale([0.2, 0.2, 0.2])
        obj.set_scale([obj_scale * 0.01, obj_scale * 0.01, obj_scale * 0.01])
        # # print(obj.get_bound_box())
        obj.enable_rigidbody(active=True)
    print('max_size:', max_size)
    print('sizes:', sizes)
    bproc.camera.set_intrinsics_from_K_matrix(
        [[577.5, 0.0, 319.5],
        [0.0, 577.5, 239.5],
        [0.0, 0.0, 1.0]], 640, 480
    )
    init_pose = np.random.uniform(np.array(room_dict[room_name][0]), np.array(room_dict[room_name][1]))
    init_pose[2] = room_dict[room_name][1][2]
    def sample_pose(obj: bproc.types.MeshObject):
        location = init_pose + np.random.uniform([-0.8, -0.8, -2], [0.8, 0.8, 2])
        if location[0] <= room_dict[room_name][0][0]:
            location[0] = room_dict[room_name][0][0]
        if location[0] >=room_dict[room_name][1][0]:
            location[0] = room_dict[room_name][1][0]
        if location[1] <= room_dict[room_name][0][1]:
            location[1] = room_dict[room_name][0][1]
        if location[1] >= room_dict[room_name][1][1]:
            location[1] = room_dict[room_name][1][1] 
        obj.set_location(location)
        print(catId_to_name[obj.get_cp("category_id")], obj.get_location())
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

    
    # Sample the poses of all spheres above the ground without any collisions in-between
    bproc.object.sample_poses(
        objs,
        sample_pose_func=sample_pose,
        max_tries = 10
    )
    
    print("Complete sampling the poses")
    # The ground should only act as an obstacle and is therefore marked passive.
    # To let the spheres fall into the valleys of the ground, make the collision shape MESH instead of CONVEX_HULL.
    room.enable_rigidbody(active=False, collision_shape="MESH")
    # Run the simulation and fix the poses of the spheres at the end
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=0.1, max_simulation_time=5, check_object_interval=1, substeps_per_frame = 1)
    print("Complete!")
    print(objs)
    poi = bproc.object.compute_poi(objs)
    cam2world_matrixs = []
    camera_scale = max_size/50/1.5
    print("camera_scale:", 0.8*roomscale*camera_scale, 1*roomscale*camera_scale)
    for i in range(pic_num):
        select = random.choice(objs)
        center = np.array(select.blender_obj.location)
        location = bproc.sampler.shell(
                center = poi,
                radius_min=0.8*roomscale*camera_scale,
                radius_max=1*roomscale*camera_scale,
                elevation_min=30,
                elevation_max=90
        ) 
        if location[0] <= room_dict[room_name][0][0]:
            location[0] = room_dict[room_name][0][0]
        if location[0] >=room_dict[room_name][1][0]:
            location[0] = room_dict[room_name][1][0]
        if location[1] <= room_dict[room_name][0][1]:
            location[1] = room_dict[room_name][0][1]
        if location[1] >= room_dict[room_name][1][1]:
            location[1] = room_dict[room_name][1][1] 
        if location[2] >= room_dict[room_name][1][2]:
            location[2] = room_dict[room_name][1][2] 
        print("Camera_loc:", location)
        if location[2] < room_dict[room_name][0][2]-20 and os.path.exists(output_dir):
            os.rmdir(output_dir)
            print(output_dir)
            print("not good!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return
                
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.0, 0.0))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
        cam2world_matrixs.append(cam2world_matrix)
        
    # 测试物体的位置
    gt_data = dict()
    category_ids = []
    instance_ids = []
    locations = []
    rotation_mats = []
    scales = []
    local2world_mats = []
    
    error_num = 0
    for obj in objs:
        category_id = obj.get_cp("category_id")
        instance_id = obj.get_cp("instance_id")
        obj_scale = obj.get_cp("obj_scale")
        location = obj.get_location()
        rotation_mat = obj.get_rotation_mat()
        scale = obj.get_scale()
        local2world_mat = obj.get_local2world_mat()
        category_ids.append(category_id)
        instance_ids.append(instance_id)
        locations.append(location)
        rotation_mats.append(rotation_mat)
        scales.append(scale)
        local2world_mats.append(local2world_mat)
        if location[2] < room_dict[room_name][0][2]-20:
            error_num += 1
    if error_num >= 3 and os.path.exists(output_dir):
        os.rmdir(output_dir)
        print(output_dir)
        return

    gt_data["category_id"] = category_ids
    gt_data["instance_id"] = instance_ids
    gt_data["location"] = locations
    gt_data["rotation_mat"] = rotation_mats
    gt_data["scale"] = scales
    gt_data["size"] = sizes
    gt_data["local2world_mat"] = local2world_mats
    gt_data["cam2world_matrix"] = cam2world_matrixs

    gt_file = os.path.join(output_dir, 'gt.pkl')
    with open(gt_file, "wb") as f:
        pickle.dump(gt_data, f)
    f.close()
    bproc.renderer.enable_segmentation_output(map_by= "instance_id", default_values={'instance_id': 255})
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    data = bproc.renderer.render()
    nocs_data = bproc.renderer.render_nocs()
    data.update(nocs_data)   
    for item in data:
        for id in range(len(data[item])):
            array = data['instance_id_segmaps'][id]
            room_area = (array == 255).sum()
            background = (array == 0).sum()
            object_area = 480*640 - room_area - background
            if background > 0 or object_area < 1000 or room_area == 0:
                continue
            meta_file = os.path.join(output_dir, str(id).zfill(4)+'_meta.txt')
            with open(meta_file, 'w', encoding='utf-8') as f:
                for obj_id in range(len(objs_path)):
                    cate = objs_path[obj_id].split(os.sep)[-3]
                    if cate.endswith("_meshes"):
                        cate = cate.replace("_meshes", "")
                    obj_folder = objs_path[obj_id].split(os.sep)[-2]
                    cate_id = name_to_catId[cate]
                    result2txt = ' '.join([str(obj_id+1), str(cate_id), cate, obj_folder]) + '\n'
                    f.write(result2txt)
            f.close()
            if item in ['colors', 'depth', 'instance_id_segmaps', 'nocs']:
                array = np.array(data[item][id])
                if item == 'colors':
                    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                    output_file = os.path.join(output_dir, str(id).zfill(4)+'_color.png')
                    cv2.imwrite(output_file, array)
                    print(room_area, background, object_area)
                elif item == 'depth':
                    output_file = os.path.join(output_dir, str(id).zfill(4)+'_depth.png')
                    array = array*10000
                    print('far:', np.max(np.max(array)))
                    array0 = array//(256*256)
                    array1 = (array//256)%256
                    array2 = array%256
                    array3 = np.ones(array.shape)
                    array0 = np.expand_dims(array0, 2)
                    array1 = np.expand_dims(array1, 2)
                    array2 = np.expand_dims(array2, 2)
                    array3 = np.expand_dims(array3, 2)
                    new = np.concatenate([array0, array1, array2, array3*255],axis = 2) 
                    cv2.imwrite(output_file, new)
                elif item == 'instance_id_segmaps':
                    output_file = os.path.join(output_dir, str(id).zfill(4)+'_mask.png')
                    cv2.imwrite(output_file, array)
                elif item == 'nocs':
                    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                    output_file = os.path.join(output_dir, str(id).zfill(4)+'_coord.png')
                    array *= 255
                    array[np.array(data['instance_id_segmaps'][id]) == 255] = np.array([255, 255, 255])
                    cv2.imwrite(output_file, array)
            else:
                continue
             

create(room_name)
