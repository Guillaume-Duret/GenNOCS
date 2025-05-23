import blenderproc as bproc
import argparse
import os
import numpy as np
from typing import List
import sys

sys.path.append('.')

# from utils.save_images import save_images

###############################################################################################################
import os
import numpy as np
from typing import List

from PIL import Image
import cv2


def save_color(img, bg: np.ndarray, index: int, out_dir: str) -> None:
    """
    Save the colored image of the scene.
    :param bg: background image.
    :param img: array representing the image.
    :param index: image shot index.
    :param out_dir: path to the folder where the image should be saved.
    """
    imgNumber = '0' * (4 - len(str(index))) + str(index)
    filename = "{}_color.png".format(imgNumber)

    front = Image.fromarray(img)
    back = Image.fromarray(bg).resize(front.size)

    back.paste(front, mask=front.convert('RGBA'))
    back.save(out_dir + '/' + filename)


def save_depth(img, index: int, out_dir: str) -> None:
    """
    Save the depth image of the scene.
    :param img: array representing the image.
    :param index: image shot index.
    :param out_dir: path to the folder where the image should be saved.
    """
    imgNumber = '0' * (4 - len(str(index))) + str(index)
    filename = "{}_depth.png".format(imgNumber)
    
    # Scale the pixel values to 16-bit
    image_array_16bit = img.astype(np.uint16) * 256
    # print(image_array_16bit.dtype)

    cv2.imwrite(out_dir + '/' + filename, image_array_16bit)



def save_nocs(img, index: int, out_dir: str) -> None:
    """
    Save the NOCS annotations image.
    :param img: array representing the NOCS image.
    :param index: image shot index.
    :param out_dir: path to the folder where the image should be saved.
    """
    imgNumber = '0' * (4 - len(str(index))) + str(index)
    # NOCS
    filename = "{}_coord.png".format(imgNumber)
    # Map image to [0;255]
    maxi = np.max(abs(img))
    img_correct = (255 - (img / maxi * 255)).astype(np.uint8)
    cv2.imwrite(out_dir + '/' + filename, cv2.cvtColor(img_correct, cv2.COLOR_RGB2BGR))


def save_mask(img, index: int, out_dir: str):
    """
    Save segmentation mask image. Color correspond to index in the meta.txt file.
    :param img: array representing the instance segmentation mask image.
    :param index: image shot index.
    :param out_dir: path to the folder where the image should be saved.
    """
    imgNumber = '0' * (4 - len(str(index))) + str(index)
    filename = "{}_mask.png".format(imgNumber)

    array = img.astype(np.uint8)
    # print(array)
    array -= 8  # Minus one for the background, minus one for the ground_obj
    array[array == -2] = 255  # Change background to white
    array = cv2.bitwise_not(array)
    cv2.imwrite(out_dir + '/' + filename, array)


def folderToCategory(folder: str) -> int:
    """
    Encode the object's category (its folder) as an integer.
    :param folder: Mesh folder.
    :return: Category number.
    """
    if folder == 'Banana_meshes':
        return 0
    elif folder == 'Broccoli_meshes':
        return 1
    elif folder == 'Cucumber_meshes':
        return 2
    elif folder == 'Pear_meshes':
        return 3
    else:
        return 10


def save_meta(obj_file: List[str], shots_number: int, out_dir: str) -> None:
    """
    Save meta information in a text file.
    :param obj_file: List of object's path in the scene
    :param shots_number: Number of files to generate.
    :param out_dir: path to the folder where the file should be saved.
    """
    file_string = ""
    for i in range(len(obj_file)):
        path = obj_file[i].split('/', 1)
        file_string += "{} {} {} {}\n".format(i, folderToCategory(path[0]), path[0], path[1])
    for index in range(shots_number):
        imgNumber = '0' * (4 - len(str(index))) + str(index)
        filename = "{}_meta.txt".format(imgNumber)
        with open(out_dir + '/' + filename, 'w') as f:
            f.write(file_string)


def save_images(images: dict, obj_files: List[str], backgrounds: List[np.ndarray], out_dir: str) -> None:
    """
    Save images and meta file.
    :param images: images rendered by blenderproc.
    :param obj_files: object files used to generate image.
    :param out_dir: path to the folder where the images should be saved.
    """
    colors = images['colors']
    for index in range(len(colors)):
        save_color(colors[index], backgrounds[index], index, out_dir)
    depth = images['depth']
    for index in range(len(depth)):
        print("hello i m herrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrre")
        save_depth(depth[index], index, out_dir)
    nocs = images['nocs']
    for index in range(len(nocs)):
        save_nocs(nocs[index], index, out_dir)
    mask = images['instance_segmaps']
    for index in range(len(mask)):
        save_mask(mask[index], index, out_dir)
    save_meta(obj_files, len(images['colors']), out_dir)



# import blenderproc as bproc
import pickle
from typing import List
import numpy as np



def getRTsMatrix(obj: bproc.types.MeshObject) -> np.ndarray:
    """
    Compute the rotation, translation and scale matrix of an object.
    :param obj: Mesh object in the scene
    :return: The RTs matrix of the object.
    """
    euler = obj.get_rotation_euler()
    translation = obj.get_location()
    scale = obj.get_scale()

    rotationX = np.matrix(
        [[1, 0, 0], [0, np.cos(euler[0]), np.sin(euler[0])], [0, -np.sin(euler[0]), np.cos(euler[0])]])
    rotationY = np.matrix(
        [[np.cos(euler[1]), 0, -np.sin(euler[1])], [0, 1, 0], [np.sin(euler[1]), 0, np.cos(euler[1])]])
    rotationZ = np.matrix(
        [[np.cos(euler[2]), -np.sin(euler[2]), 0], [np.sin(euler[2]), np.cos(euler[2]), 0], [0, 0, 1]])
    rotation = rotationX * rotationY * rotationZ

    rotation_mat = np.matrix(
        np.vstack((
            np.hstack((
                rotation, np.zeros((3,1))
            )),
            np.array([0, 0, 0, 1]))
        )
    )
    translation_mat = np.matrix([
        [1, 0, 0, translation[0]],
        [0, 1, 0, translation[1]],
        [0, 0, 1, translation[2]],
        [0, 0, 0, 1]
    ])
    scale_mat = np.matrix([
        [scale[0], 0, 0, 0],
        [0, scale[1], 0, 0],
        [0, 0, scale[2], 0],
        [0, 0, 0, 1]
    ])
    rts_mat = rotation_mat * scale_mat * translation_mat

    return np.squeeze(np.asarray(rts_mat))


def getBboxOnImage(img: np.ndarray, instance: int) -> List[int]:
    """
    Find the image bounding box of a specific instance.
    :param img: The mask image as an array.
    :param instance: The instance to loo for.
    :return: An array describing the bounding box as y_min, x_min, y_max, x_max.
    """
    x, y = np.where(img == instance)
    x_min = np.min(x)
    y_min = np.min(y)
    x_max = np.max(x)
    y_max = np.max(y)
    return [x_min, y_min, x_max, y_max]


def save_gts(objs: List[bproc.types.MeshObject], mesh_files: List[str], im_output_dir: str, scene_id: str,
             im_number: int, mask: np.ndarray, output_dir: str):
    rts = np.array([getRTsMatrix(obj) for obj in objs])
    instances = set(mask.flatten())
    # Remove white background
    instances.discard(0)
    instances.discard(1)
    bboxes = np.array([getBboxOnImage(mask, i) for i in instances])
    classes = np.array([folderToCategory(file_path.split('/', 1)[0]) for file_path in mesh_files])
    scales = np.array([obj.get_scale() for obj in objs])
    im_id = scene_id.lstrip('0') + str(im_number)
    im_number_string = '0' * (4 - len(str(im_number))) + str(im_number)
    im_path = os.path.join(im_output_dir, im_number_string)

    dico = {
        'gt_RTs': rts,
        'gt_bboxes': bboxes,
        'gt_class_ids': classes,
        'gt_scales': scales,
        'image_id': im_id,
        'image_path': im_path,
        'obj_list': mesh_files
    }

    filename = 'results_val_{}_{}.pkl'.format(scene_id, im_number_string)
    save_path = os.path.join(output_dir, filename)
    file = open(save_path, 'wb')
    pickle.dump(dico, file)
    file.close()











##################################################################################################################
from utils.background_and_camera import pick_random_background, load_background, get_camera_position_from_background, \
    get_table_dim
# from utils.save_gts import save_gts


def create_scene(mesh_path: str, ground_obj: bproc.types.MeshObject, json_file_path: str) -> (
        List[bproc.types.MeshObject], List[str]):
    """
    Loads random object mesh at random positions into space. Physics simulation assure objects 'fall' to a plane at z=0.
    :param mesh_path: folder where mesh objects are stored.
    :param ground_obj: mesh object used in simulation to ensure loaded objects end up on the z=0 plane.
    :return: A list of the loaded objects and a list of their local paths from mesh_path.
    """
    # Load random objects
    number_of_objects = 6

    objects = []
    objects_files = []

    for _ in range(number_of_objects):
        path = mesh_path
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        while len(files) == 0:  # Navigate to a random folder
            folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            path = os.path.join(path, folders[np.random.randint(0, len(folders))])
            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        folder_relative_path = path.replace(mesh_path, '').replace('\\', '/')
        if folder_relative_path.startswith('/'):
            folder_relative_path = folder_relative_path[1:]
        objects_files.append(folder_relative_path)

        # Load mesh to scene
        # obj_file = os.path.join(path, [f for f in files if f[-4:] == ".obj"][0])
        obj_files = [f for f in files if f.endswith('.obj')]
        if obj_files:
            random_obj_file = np.random.choice(obj_files)
            obj_file_path = os.path.join(path, random_obj_file)
            try:
                # Load mesh to scene
                objects.append(bproc.loader.load_obj(obj_file_path)[0])
            except Exception as e:
                print(f"Error loading {obj_file_path}: {e}")
        else:
            print(f"No .obj files found in {path}")
        # objects.append(bproc.loader.load_obj(obj_file)[0])


    table_dim = get_table_dim(json_file_path) * 0.35
    low = [-table_dim[0]/2 , -table_dim[1]/2 , 1]
    high = [table_dim[0]/2 , table_dim[1]/2 , 2]
    # print(low,high)

    room_planes = [bproc.object.create_primitive('PLANE', scale=[20, 20, 10]),
               bproc.object.create_primitive('PLANE', scale=[20, 20, 10], location=[0, -20, 20], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[20, 20, 10], location=[0, 20, 20], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[20, 20, 10], location=[20, 0, 20], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[20, 20, 10], location=[-20, 0, 20], rotation=[0, 1.570796, 0])]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)





    # Place randomly object without collision
    def sample_pose(obj: bproc.types.MeshObject):
        obj.set_rotation_euler(bproc.sampler.uniformSO3())
        # current_rotation = obj.get_rotation_euler()
        obj.set_scale(np.random.uniform([0.8, 0.8, 0.8], [1.2, 1.2, 1.2]))
        obj.set_location(np.random.uniform(low, high))
        # Restore the original rotation

    
    bproc.object.sample_poses(
        objects,
        sample_pose_func=sample_pose
    )

    # Make all objects actively participate in the simulation
    for obj in objects:
        obj.enable_rigidbody(active=True)
    # The ground should only act as an obstacle and is therefore marked passive.
    ground_obj.enable_rigidbody(active=False, collision_shape="CONVEX_HULL")

    # Run the simulation and fix the poses of the objets at the end
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=20,
                                                      check_object_interval=1)

    # Remove (hide) ground
    ground_obj.set_location([0, 0, 200])

    return objects, objects_files


def create_static_scene(mesh_path: str, ground_obj: bproc.types.MeshObject) -> (
        List[bproc.types.MeshObject], List[str]):
    """
    Loads specific object mesh at specific positions into space. Physics simulation assure objects 'fall' to a plane at
    z=0.
    :param mesh_path: folder where mesh objects are stored.
    :param ground_obj: mesh object used in simulation to ensure loaded objects end up on the z=0 plane.
    :return: A list of the loaded objects and a list of their local paths from mesh_path.
    """
    # Load random objects

    objects = []
    objects_files = ['Banana_meshes/01/banana01.obj', 'Mandarine_meshes/01/mandarine01.obj',
                     'Pear_meshes/01/pear01.obj']

    for file in objects_files:
        obj_file = os.path.join(mesh_path, file)
        # Load mesh to scene
        objects.append(bproc.loader.load_obj(obj_file)[0])

    objects[0].set_location([0, 0, 1])
    objects[1].set_location([2, 0, 1])
    objects[2].set_location([0, 2, 1])

    # Make all objects actively participate in the simulation
    for obj in objects:
        obj.enable_rigidbody(active=True)
    # The ground should only act as an obstacle and is therefore marked passive.
    ground_obj.enable_rigidbody(active=False, collision_shape="CONVEX_HULL")

    # Run the simulation and fix the poses of the objets at the end
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=20,
                                                      check_object_interval=1)

    # Remove (hide) ground
    ground_obj.set_location([0, 0, 30])

    return objects, objects_files


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help="Absolute path to the data folder where images are save, mesh are stored...")
    parser.add_argument("--background", required=True, help="Path to background")
    parser.add_argument('--shots_number',required=True, help="Number of image of the scene to generate")
    parser.add_argument('--mode',required=True, help="Can be either train or val")
    args = parser.parse_args()

    bproc.init()

    # Pick background
    if os.path.isdir(args.background):
        bg_file = pick_random_background(args.background)
    else:
        bg_file = args.background
    json_path = os.path.splitext(bg_file)[0] + ".json"

    
    # Load objects
    ground = bproc.loader.load_obj("/home/ybourennane/blenderProc/shapo-main/blenderproc/ground.obj")[0]
    mesh_dir =  os.path.join(args.data_dir, 'obj_models', args.mode)
    # print(mesh_dir)
    objs, files = create_scene(mesh_dir, ground, json_path)

    # define a light and set its location and energy level
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(1000)
    
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([-5, 5, 5])
    light.set_energy(1000)

    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([-5, -5, 5])
    light.set_energy(1000)

    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, 5, 5])
    light.set_energy(1000)

    
    # define the camera resolution
    bproc.camera.set_resolution(640, 480)
    bg_images = []

    for _ in range(int(args.shots_number)):
        # Sample random camera location around the object using background image
        bg_img, frame_number = load_background(bg_file)
        bg_images.append(bg_img)

        cam_location, img_center_pos = get_camera_position_from_background(bg_img, json_path, frame_number)
        if cam_location is None:  # We could not estimate the cam position, just skip
            print("Frame skipped")
            continue
        rotation_matrix = bproc.camera.rotation_from_forward_vec(img_center_pos - cam_location)
        # Add homog cam pose based on location and rotation
        cam2world_matrix = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)


    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_output_format(enable_transparency=True)

    # render the whole pipeline
    data = bproc.renderer.render()
    data.update(bproc.renderer.render_nocs())
    data.update(bproc.renderer.render_segmap(map_by="instance", render_colorspace_size_per_dimension=len(objs) + 10))

        # Save data
    camera_dir = os.path.join(args.data_dir, 'CAMERA', args.mode)
    if not os.path.exists(camera_dir):
        os.makedirs(camera_dir, exist_ok=True)
    folders = [f for f in os.listdir(camera_dir) if os.path.isdir(os.path.join(camera_dir, f))]

    if len(folders) == 0:
        folder_name = '0' * 5
    else:
        folder_num = int(sorted(folders)[-1]) + 1
        folder_name = '0' * (5 - len(str(folder_num))) + str(folder_num)

    true_out_dir = os.path.join(camera_dir, folder_name)

    os.makedirs(true_out_dir, exist_ok=True)


    save_images(data, files, bg_images, true_out_dir)

    if args.mode == 'val':  # Scene is for the validation set
        gts_dir = os.path.join(args.data_dir, 'gts', 'val')
        if not os.path.exists(gts_dir):
            os.makedirs(gts_dir, exist_ok=True)

        for index in range(len(data['instance_segmaps'])):
            save_gts(objs, files, true_out_dir, folder_name, index, data['instance_segmaps'][index], gts_dir)

