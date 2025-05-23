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
parser.add_argument('--scene', default="/home/mazurrrak/ECL/Semester2/PO/Omni6D/data/replica", help="Path to where the final files, will be saved, could be examples/basics/basic/output")
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

pic_num =2
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
    catId_to_name = {1: 'bottle', 2: 'bowl', 3: 'camera', 4: 'can', 5: 'laptop2', 6: 'mug', 7: 'apples', 8: 'apricots', 9: 'bananas', 10: 'dragon', 11: 'figs', 12: 'oranges', 13: 'peaches', 14: 'pears', 15: 'pineapples', 16: 'watermelons', 17: 'artichokes', 18: 'broccoli', 19: 'cauliflower', 20: 'corn', 21: 'jalapeño', 22: 'lettuce', 23: 'mushrooms', 24: 'potatoes', 25: 'pumpkin', 26: 'tomatoes'}
else:
    catId_to_name = {1: 'bottle', 2: 'bowl', 3: 'camera', 4: 'can', 5: 'laptop2', 6: 'mug', 7: 'apples', 8: 'apricots', 9: 'bananas', 10: 'dragon', 11: 'figs', 12: 'oranges', 13: 'peaches', 14: 'pears', 15: 'pineapples', 16: 'watermelons', 17: 'artichokes', 18: 'broccoli', 19: 'cauliflower', 20: 'corn', 21: 'jalapeño', 22: 'lettuce', 23: 'mushrooms', 24: 'potatoes', 25: 'pumpkin', 26: 'tomatoes'}

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
    
    model_objects = "/home/mazurrrak/ECL/Semester2/PO/Omni6D/data/obj_models_small/"
    model_num = random.choice(list(range(4,7)))
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

    global_scale = 3
    scale_ranges = {
        'bottle': ([15.0, 15.0, 15.0], [30.0, 30.0, 30.0]),
        'bowl': ([10.0, 10.0, 10.0], [15.0, 15.0, 15.0]),
        'camera': ([10.0, 10.0, 10.0], [15.0, 15.0, 15.0]),
        'can': ([7.0, 7.0, 7.0], [12.0, 12.0, 12.0]),
        'laptop2': ([20.0, 20.0, 20.0], [30.0, 30.0, 30.0]),
        'mug': ([8.0, 8.0, 8.0], [12.0, 12.0, 12.0]),
    }

    objs = []
    for obj_id in range(len(objs_path)):
        obj_path = objs_path[obj_id]
        obj_name = os.path.basename(obj_path).lower()
        print("obj_name!!!!!!!!!!!!!!!!!!", obj_name)
        found = False
        for category, (min_scale, max_scale) in scale_ranges.items():
            if category in obj_name:
                scaled_min = np.array(min_scale) * global_scale
                scaled_max = np.array(max_scale) * global_scale
                obj_scale = np.random.uniform(scaled_min, scaled_max)
                found = True
                break
        if not found:
            raise ValueError(f"No matching category found in object name '{obj_name}'. Ensure the object name includes one of: {list(scale_ranges.keys())}.")
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

    light = bproc.types.Light()
    '''点光源的设置'''
    light_class = 'POINT'
    light.set_energy(random.uniform(500, 700)*roomscale)
    if room_name == 'room_0':
        light.set_energy(random.uniform(550, 650)*roomscale)
    if room_name == 'room_1':
        light.set_energy(random.uniform(400, 500)*roomscale)
    if room_name == 'hotel_0':
        light.set_energy(random.uniform(1100, 1300)*roomscale)
    if room_name == 'office_2':
        light.set_energy(random.uniform(1400, 1600)*roomscale)
    if room_name == 'office_3':
        light.set_energy(random.uniform(550, 650)*roomscale)
    light.set_type(light_class)
    center_init = np.mean(np.array([room_dict[room_name][0], room_dict[room_name][1]]), axis = 0)
    center_init[2] = room_dict[room_name][1][2]
    light.set_location(bproc.sampler.shell(
        center = center_init,
        radius_min=0.4*roomscale,
        radius_max=0.6*roomscale,
        elevation_min=0,
        elevation_max=1
    ))
    
    sizes = []
    max_size = 0
    for obj in objs:
        obj_scale = obj.get_cp("obj_scale")
        obj_scale = float(obj_scale[0])
        size = obj.get_bound_box()
        size_x = max(abs(size[:, 0]))
        size_y = max(abs(size[:, 1]))
        size_z = max(abs(size[:, 2]))
        if obj_scale > max_size:
            max_size = obj_scale
        real_size = [size_x, size_y, size_z, obj_scale * 0.02]
        sizes.append(real_size)
        # 前者表示归一化模型的scale，后者表示模型从[0,1]^3单位空间中的放缩
        # obj.set_scale([0.2, 0.2, 0.2])
        obj.set_scale([obj_scale * 0.02, obj_scale * 0.02, obj_scale * 0.02])
        # # print(obj.get_bound_box())
        obj.enable_rigidbody(active=True)
    print('max_size:', max_size)
    print('sizes:', sizes)
    bproc.camera.set_intrinsics_from_K_matrix(
        [[577.5, 0.0, 319.5],
        [0.0, 577.5, 239.5],
        [0.0, 0.0, 1.0]], 640, 480
    )

    bproc.camera.set_stereo_parameters(convergence_mode="PARALLEL", convergence_distance=10.0, interocular_distance=0.05)
    
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
    print("poi:", poi)
    cam2world_matrixs = []
    camera_scale = max_size/50/1.5
    print("max_size", max_size)
    print("camera_scale:", 0.8*roomscale*camera_scale, 1*roomscale*camera_scale)
    for i in range(pic_num):
        select = random.choice(objs)
        center = np.array(select.blender_obj.location)
        print(0.8*roomscale*camera_scale, 1*roomscale*camera_scale)
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
        print("location", location[2])
        print("room_dict", room_dict[room_name][0][2]-20)
        print("os.path.exists(output_dir)", os.path.exists(output_dir))
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
        # base_cam_pose = bproc.math.build_transformation_mat(location, rotation_matrix)

        # interocular_distance = 0.01

        # offset = interocular_distance / 2.0

        # left_offset = np.array([-offset, 0, 0])
        # right_offset = np.array([ offset, 0, 0])

        # left_cam_pose = base_cam_pose.copy()
        # right_cam_pose = base_cam_pose.copy()

        # left_cam_pose[:3, 3] = left_cam_pose[:3, 3] + left_cam_pose[:3, :3].dot(left_offset)
        # right_cam_pose[:3, 3] = right_cam_pose[:3, 3] + right_cam_pose[:3, :3].dot(right_offset)

        # bproc.camera.add_camera_pose(left_cam_pose)
        # bproc.camera.add_camera_pose(right_cam_pose)
        # cam2world_matrixs.append(left_cam_pose)
        # cam2world_matrixs.append(right_cam_pose)
        
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

    bproc.renderer.toggle_stereo(True)

    bproc.renderer.enable_segmentation_output(map_by= "instance_id", default_values={'instance_id': 255})
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    data = bproc.renderer.render()
    nocs_data = bproc.renderer.render_nocs()
    data.update(nocs_data)   
    print("Data keys:", data.keys())
    print("Number of images in data['colors']:", len(data["colors"]))
    for idx, color_img in enumerate(data["colors"]):
        color_array = np.array(color_img)
        print(f"data['colors'][{idx}] shape: {color_array.shape}, min: {color_array.min()}, max: {color_array.max()}")

    print("data[colors][0].shape :", data["colors"][0].shape) 
    # print("data[colors][1].shape :",data["colors"][1].shape)

    # if len(data["colors"]) % 2 != 0:
    #     print("Warning: number of color images is not even; some stereo pair might be incomplete!")
    # stereo_pairs = []
    # for i in range(0, len(data["colors"]), 2):
    #     left = np.array(data["colors"][i])
    #     right = np.array(data["colors"][i+1])
    #     pair = np.stack([left, right], axis=0)
    #     stereo_pairs.append(pair)

    data["stereo-depth"], data["disparity"] = bproc.postprocessing.stereo_global_matching(data["colors"], window_size = 15, num_disparities = 96, 
                                                                                min_disparity = 0, disparity_filter= True, depth_completion = True)
    
    # print("data[colors][0].shape :", data["colors"][0].shape) 
    # print("data[colors][1].shape :",data["colors"][1].shape)
    
    # print("colors :", data['colors'][0].shape)
    # print("depth :", data['depth'][0].shape)
    # print("stereo-depth :", data['stereo-depth'][0].shape)
    # print("disparity :", data['disparity'][0].shape)
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
            if item in ['colors', 'depth', 'stereo-depth', 'instance_id_segmaps', 'nocs']:
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

                    output_file = os.path.join(output_dir, str(id).zfill(4) + '_depth_norm.png')
                    depth_norm = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                    cv2.imwrite(output_file, depth_color)
                elif item == 'stereo-depth':
                    output_file = os.path.join(output_dir, str(id).zfill(4)+'_stereo_depth.png')
                    array = np.array(data[item][id]) * 10000
                    print('stereo far:', np.max(np.max(array)))
                    array0 = array // (256 * 256)
                    array1 = (array // 256) % 256
                    array2 = array % 256
                    array3 = np.ones(array.shape)
                    array0 = np.expand_dims(array0, 2)
                    array1 = np.expand_dims(array1, 2)
                    array2 = np.expand_dims(array2, 2)
                    array3 = np.expand_dims(array3, 2)
                    new = np.concatenate([array0, array1, array2, array3 * 255], axis=2)
                    cv2.imwrite(output_file, new)
                    output_file = os.path.join(output_dir, str(id).zfill(4) + '_stereo_depth_norm.png')
                    stereo_norm = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    stereo_color = cv2.applyColorMap(stereo_norm, cv2.COLORMAP_JET)
                    cv2.imwrite(output_file, stereo_color)

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
