import blenderproc as bproc
import argparse
import numpy as np
import cv2
import os
import sys
import json
import random
from blenderproc.python.loader.HavenEnvironmentLoader import set_world_background_hdr_img, get_random_world_background_hdr_img_path_from_haven
from PIL import Image
import pickle
import glob
import math
from math import atan, tan
import numpy as np
import mathutils
from mathutils import Vector, Euler, Matrix
import bpy


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default="/home/mazurrrak/ECL/Semester2/PO/Omni6D/output", help="Path to where the final files, will be saved, could be examples/basics/basic/output")
parser.add_argument('--split', default="train", help="dataset split you want to generate, could be train, val, test, test_unseen")
parser.add_argument('--scene', default="/home/mazurrrak/ECL/Semester2/PO/Omni6D/data/ikea_data", help="Path to the scene dataset")
parser.add_argument('--objects', default="/home/mazurrrak/ECL/Semester2/PO/Omni6D/data/obj_meshes_NOCS/", help="Path to the object dataset")
parser.add_argument('--setShadows', default=True, help="Set to True to enable shadows")

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

pic_num = 100

roomscale = 10
room_dict = {'table1': [[0.0135, 1.74749994, 0.0075], [2.83890009, 2.9625001, 2.99250007]],
             'table2': [[0.0057, 1.13730001, 0.0057], [2.08660007, 2.24580002, 2.26850009]],
             'table3': [[0.0028, 0.38929999, 0.0029], [1.1832, 1.25090003, 1.26349998]],
             'table4': [[0.0029, 0.41150001, 0.0029], [1.00450003, 1.15869999, 1.17040002]],
             'table5': [[0.1436, 0.66149998, 0.19490001], [1.06210005, 1.11310005, 1.12440002]],
             'table6': [[0.0029, 0.0364, 0.0029], [1.17490005, 1.1631, 1.17490005]],
             'table7': [[0.0075, 1.52250004, 0.0075], [2.99250007, 2.9625001, 1.51240003]],
             'table8': [[0.0075, 1.50750005, 0.0075], [2.99250007, 2.9625001, 1.91250002]],
             'table9': [[0.0075, 1.23749995, 0.0075], [2.99250007, 2.9625001, 2.99250007]],
             'table10': [[0.0067, 0.99650002, 0.0067], [2.6487999, 2.64170003, 2.66849995]],
             'table11': [[0.0043, 0.0043, 0.0043], [1.65289998, 1.69949996, 1.71669996]],
             'table12': [[0.0075, 1.50750005, 0.0075], [2.99250007, 2.9625001, 2.99250007]],
             'table13': [[0.0043, 0.59460002, 0.3423], [1.70899999, 1.69190001, 1.43210006]],
             'table14': [[0.0651, 0.1374, 0.0072], [2.30660009, 1.64139998, 2.08450007]],
             'table15': [[0.0041, 0.0041, 0.0041], [1.64139998, 1.62489998, 1.56729996]],
             'table16': [[0.0062, 0.5187, 0.0062], [2.26230001, 2.46869993, 2.19860005]],
             'table17': [[0.0056, 0.84359998, 0.0056], [2.22919989, 2.20679998, 2.22919989]],
             'table18': [[0.0062, 0.9109, 0.0062], [2.47239995, 2.44770002, 2.29049993]],
             'table19': [[0.0062, 0.90920001, 0.0062], [2.46070004, 2.43610001, 2.46070004]],
             'table20': [[0.0075, 1.3125, 0.19850001], [2.9223001, 2.9625001, 1.79550004]],
             'table21': [[0.0073, 1.71560001, 0.0073], [2.90229988, 2.89120007, 2.92050004]],
             'table22': [[0.0058, 0.58840001, 0.0058], [2.32450008, 2.30119991, 2.32450008]],
             'table23': [[0.0032, 0.1392, 0.0032], [1.29120004, 1.27830005, 1.20060003]],
             'table25': [[0.0043, 0.9429, 0.0043], [1.61510003, 1.68990004, 1.70710003]],
             'table26': [[0.2834, 1.18470001, 0.0056], [2.2191999, 2.19700003, 2.2191999]],
             'table27': [[0.0045, 0.99080002, 0.0045], [1.77670002, 1.75890005, 1.77670002]],
             'table28': [[0.0058, 0.3969, 0.1323], [2.29480004, 2.27180004, 2.24900007]],
             'table29': [[0.0048, 0.3673, 0.0048], [1.90339994, 1.88429999, 1.45850003]],
             'table30': [[0.0045, 0.7493, 0.0045], [1.70299995, 1.7938, 1.81200004]],
             'table31': [[0.25080001, 0.65990001, 0.0033], [1.32309997, 1.30980003, 1.32309997]],}

if split == 'test_unseen':
    catId_to_name = {1: 'bottle', 2: 'bowl', 3: 'camera', 4: 'can', 5: 'laptop', 6: 'mug'}
    # catId_to_name = {1001: 'air_purifier', 1002: 'apricot', 1003: 'award', 1004: 'ax', 1005: 'bamboo_flute', 1006: 'baseball_bat', 1007: 'baseball_glove', 1008: 'battery', 1009: 'beachball', 1010: 'beauty_blender', 1011: 'bed', 1012: 'binder', 1013: 'blackboard_eraser', 1014: 'blueberry', 1015: 'bookmark', 1016: 'boomerang', 1017: 'bowling_ball', 1018: 'bumbag', 1019: 'cabinet', 1020: 'camel', 1021: 'candied_haws', 1023: 'canned_beverage', 1024: 'chestnut', 1025: 'chinese_cabbage', 1026: 'cigar', 1027: 'clothespin', 1028: 'corkscrew', 1029: 'croquet', 1030: 'ear_phone', 1031: 'electric_cake', 1032: 'electric_pressure_cooker', 1033: 'facial_cleaner', 1035: 'faucet', 1036: 'foot_bath', 1037: 'fork', 1038: 'frisbee', 1039: 'fruit_basket', 1040: 'funnel', 1041: 'glasses', 1042: 'golf', 1043: 'gong', 1044: 'gravy_boat', 1045: 'guitar', 1046: 'hair_curler', 1047: 'headphone', 1048: 'high_heel', 1049: 'hockey', 1050: 'honey_dipper', 1051: 'hulusi', 1053: 'kennel', 1054: 'laptop', 1055: 'life_buoy', 1056: 'lipped', 1057: 'loafer', 1058: 'lollipop', 1059: 'magnet', 1060: 'microphone', 1061: 'microwaveoven', 1063: 'monitor', 1064: 'muffin', 1065: 'nutcracker', 1066: 'ornaments', 1067: 'paintbox', 1068: 'peach_bun', 1069: 'pear', 1070: 'pen', 1071: 'penguin', 1072: 'perfume', 1073: 'phone', 1074: 'piccolo', 1075: 'pigeon', 1076: 'plug', 1077: 'pomelo', 1078: 'pressure_cooker', 1079: 'puffed_food', 1080: 'reptiles', 1081: 'sea_horse', 1082: 'shakuhachi', 1083: 'shower_head', 1084: 'skateboard', 1085: 'slippers', 1086: 'smoke_detector', 1087: 'snare_drum', 1088: 'soccer', 1089: 'soymilk_machine', 1090: 'sweeping_robot', 1091: 'sword_bean', 1092: 'tennis_ball', 1093: 'thermometer', 1094: 'thermos', 1095: 'tobacco_pipe', 1096: 'triangle', 1097: 'tvstand', 1098: 'ukri', 1099: 'wall_lamp', 1100: 'water_chestnut', 1101: 'weaving_basket', 1102: 'wireless_walkie_talkie', 1104: 'wooden_sword', 1105: 'xylophone'}
else:
    catId_to_name = {1: 'bottle', 2: 'bowl', 3: 'camera', 4: 'can', 5: 'laptop', 6: 'mug'}
    # catId_to_name = {1: 'aerosol_can', 2: 'alligator', 3: 'almond', 4: 'animal', 5: 'anise', 6: 'antique', 7: 'apple', 8: 'asparagus', 9: 'baby_chinese_cabbage', 10: 'backpack', 11: 'bagel', 12: 'ball', 13: 'balloon', 14: 'bamboo_carving', 15: 'bamboo_shoots', 16: 'banana', 17: 'baseball', 18: 'basketball', 19: 'beer_can', 20: 'bell', 21: 'belt', 22: 'billiards', 23: 'bird', 24: 'birdbath', 25: 'birdhouse', 26: 'biscuit', 27: 'book', 28: 'bottle', 29: 'bottle_opener', 30: 'bowl', 31: 'bowling_pin', 32: 'box', 33: 'boxed_beverage', 34: 'boxing_glove', 35: 'bracelet', 36: 'bread', 37: 'broad_bean', 38: 'broccoli', 39: 'broccolini', 40: 'bronze', 41: 'brush', 42: 'brussels_sprout', 43: 'bucket_noodle', 44: 'bun', 45: 'burrito', 46: 'cabbage', 47: 'cactus', 48: 'cake', 49: 'calculator', 50: 'can', 51: 'candle', 52: 'beet', 53: 'candy_bar', 54: 'carambola', 55: 'carrot', 56: 'castanets', 57: 'cat', 58: 'cauliflower', 59: 'chair', 60: 'charcoal_carving', 61: 'cheese', 62: 'cherry', 63: 'chess', 64: 'chicken_leg', 65: 'chili', 66: 'china', 67: 'chinese_chess', 68: 'chinese_knot', 69: 'chinese_pastry', 70: 'chocolate', 72: 'cigarette', 73: 'cigarette_case', 74: 'clock', 76: 'coconut', 77: 'coin', 79: 'conch', 80: 'coral', 81: 'corn_skin', 82:'corn', 83: 'crab', 84: 'crayon', 85: 'cricket', 86: 'cucumber', 87: 'cup', 88: 'cushion', 89: 'date_fruit', 90: 'dental_carving', 91: 'desk_lamp', 92: 'dice', 93: 'dinosaur', 94: 'dish', 95: 'dishtowel', 96: 'dog', 97: 'dog_food', 98: 'doll', 99: 'donut', 100: 'drawing', 101: 'drum', 102: 'drumstick', 103: 'duct_tape', 104: 'dumbbell', 105: 'dumpling', 106: 'durian', 107: 'dustbin', 108: 'earplug', 109: 'edamame', 110: 'egg', 111: 'egg_roll', 112: 'egg_tart', 113: 'eggplant', 114: 'electric_clippers', 115: 'electric_toothbrush', 116: 'eleocharis_dulcis', 118: 'facial_cream', 119: 'facial_tissue_holder', 120: 'fan', 121: 'fig', 122: 'fire_extinguisher', 123: 'fish', 124: 'flash_light', 125: 'flower_pot', 126: 'flute', 127: 'football', 128: 'fountain_pen', 129: 'french_chips', 130: 'garage_kit', 133: 'glasses_case', 134: 'gloves', 135: 'gourd', 136: 'grass_roller', 137: 'green_bean_cake', 138: 'guacamole', 139: 'gumbo', 140: 'gundam_model', 141: 'hair_dryer', 142: 'hairpin', 143: 'hamburger', 144: 'hami_melon', 145: 'hammer', 146: 'hand_cream', 147: 'hand_drum', 148: 'handbag', 149: 'handball', 150: 'handstamp', 151: 'hat', 152: 'haw_thorn', 154: 'headband', 155: 'helmet', 156: 'horse', 157: 'hot_dog', 158: 'house', 159: 'household_blood_glucose_meter', 160: 'humidifier', 163: 'insole', 167: 'joystick', 168: 'kettle', 169: 'keyboard', 170: 'kidney_beans', 171: 'kite', 172: 'kiwifruit', 173: 'knife', 174: 'lacquerware', 175: 'lantern', 176: 'laundry_detergent', 177: 'lego', 178: 'lemon', 179: 'lettuce', 180: 'light', 181: 'lint_remover', 182: 'lipstick', 183: 'litchi', 184: 'lizard', 185: 'longan', 186: 'loquat', 187: 'lotus_root', 188: 'macaron', 190: 'mangosteen', 191: 'maracas', 192: 'mask', 193: 'matchbox', 194: 'medicine_bottle', 195: 'milk', 196: 'mooncake', 197: 'mouse', 198: 'mousepad', 199: 'mushroom', 200: 'nailfile', 201: 'nesting_dolls', 202: 'nipple', 203: 'nuclear_carving', 204: 'onion', 205: 'orange', 206: 'ornament', 207: 'oyster', 208: 'package', 209: 'pad', 210: 'padlock', 211: 'paintbrush', 212: 'pan', 213: 'pancake', 214: 'paper_knife', 215: 'passion_fruit', 216: 'pastry', 217: 'peach', 218: 'peanut', 219: 'persimmon', 220: 'phone_case', 221: 'photo_frame', 222: 'picnic_basket', 223: 'pie', 224: 'pillow', 225: 'pineapple', 226: 'pinecone', 227: 'pingpong', 228: 'pistachio', 229: 'pitaya', 230: 'pizza', 231: 'plant', 232: 'pocket_watch', 234: 'poker', 235: 'pomegranate', 236: 'popcorn', 239: 'pottery', 240: 'power_strip', 241: 'projector', 242: 'prune', 243: 'puff', 244: 'pumpkin', 245: 'puppet', 246: 'radio', 247: 'radish', 248: 'razor', 249: 'red_cabbage', 250: 'red_jujube', 251: 'red_wine_glass', 252: 'remote_control', 254: 'ricecooker', 255: 'root_carving', 256: 'rubik_cube', 257: 'sandwich', 258: 'sausage', 259: 'scissor', 260: 'screwdriver', 261: 'set_top_box', 262: 'shampoo', 263: 'shaomai', 264: 'shoe', 265: 'shrimp', 266: 'soap', 267: 'sofa', 268: 'softball', 269: 'spanner', 270: 'speaker', 271: 'spice_mill', 272: 'squash', 273: 'starfish', 274: 'steamed_bun', 275: 'steamed_twisted_roll', 276: 'stone_carving', 277: 'stool', 278: 'straw', 279: 'strawberry', 280: 'suitcase', 281: 'sushi', 282: 'sweet_potato', 283: 'table', 284: 'table_tennis_bat', 285: 'tamarind', 286: 'tambourine', 287: 'tang_sancai', 288: 'tank', 289: 'tape_measure', 290: 'taro', 291: 'teacup', 292: 'teakettle', 293: 'teapot', 294: 'teddy_bear', 295: 'thermos_bottle', 296: 'thimble', 297: 'timer', 298: 'tissue', 299: 'tomato', 300: 'tongs', 301: 'toolbox', 302: 'tooth_brush', 303: 'tooth_paste', 304: 'toothpick_box', 305: 'toy_animals', 306: 'toy_boat', 307: 'toy_bus', 308: 'toy_car', 309: 'toy_gun', 310: 'toy_motorcycle', 311: 'toy_plane', 312: 'toy_plant', 313: 'toy_train', 314: 'toy_truck', 315: 'toys', 316: 'tray', 317: 'turtle', 319: 'umbrella', 320: 'vase', 321: 'vine_ball', 322: 'volleyball', 323: 'waffle', 324: 'wallet', 325: 'walnut', 326: 'watch', 327: 'water_gun', 328: 'watering_can', 329: 'watermelon', 330: 'wenta_walnut', 331: 'whistle', 334: 'wooden_ball', 335: 'wooden_spoon', 336: 'woodfish', 337: 'world_currency', 338: 'yam', 339: 'zongzi'}

if split == 'train':
    room_list = ['table1',  'table3', 'table4', 'table5', 'table6', 'table7', 'table8', 'table9', 'table10',
                 'table12', 'table13', 'table14', 'table15', 'table16', 'table17', 'table18', 'table19',
                 'table20',  'table22', 'table23', 'table25', 'table26', 'table28', 'table29',
                 'table30', 'table31']
elif split == 'val':
    room_list = ['table2', 'table11', 'table21', 'table27']
else:
    room_list = ['table2', 'table11', 'table21', 'table27']
name_to_catId = {value:key for key,value in catId_to_name.items()}

room_name = random.choice(room_list)
print("room_name:", room_name)

def load_ikea(ikea_root: str, room_name: str, output_dir: str, use_smooth_shading: bool = True, pic_num: int = None):
    scene_folder = os.path.join(ikea_root, room_name)
    if not os.path.exists(scene_folder):
        raise FileNotFoundError(f"Scene folder {scene_folder} does not exist.")
    mesh_folder = os.path.join(ikea_root, "Meshes")
    print("mesh_folder : ", mesh_folder)
    if not os.path.exists(mesh_folder):
        raise FileNotFoundError(f"Mesh folder {mesh_folder} does not exist.")
    
    obj_files = glob.glob(os.path.join(mesh_folder, f"{room_name}.obj"))
    meshes = []
    for obj_file in obj_files:
        objs = bproc.loader.load_obj(obj_file)
        if use_smooth_shading:
            for o in objs:
                o.set_shading_mode("SMOOTH")
        meshes.extend(objs)
    
    image_paths = sorted(glob.glob(os.path.join(scene_folder, "*_color.png")))
    
    meta_file = os.path.join(scene_folder, "meta.txt")
    with open(meta_file, 'r') as f:
        lines = f.readlines()

    cam_matrices = []
    with open(meta_file, 'r') as f:
        buffer = []
        for line in f.readlines()[8:]:
            line_stripped = line.strip()
            if line_stripped:
                buffer.append(line_stripped)
            elif buffer:
                cam_matrices.append(Matrix([list(map(float, row.split())) for row in buffer]))
                buffer = []
        if buffer:
            cam_matrices.append(Matrix([list(map(float, row.split())) for row in buffer]))
    
    if len(cam_matrices) != len(image_paths):
        print(f"WARNING: Matrices count ({len(cam_matrices)}) != images count ({len(image_paths)})")
        min_count = min(len(cam_matrices), len(image_paths))
        cam_matrices = cam_matrices[:min_count]
        image_paths = image_paths[:min_count]
    else:
        print(f"Matrices and images count match: {len(cam_matrices)}")

    if pic_num is not None:
        if pic_num <= 0:
            raise ValueError("pic_num must be positive")
        total_available = len(cam_matrices)
        if pic_num > total_available:
            print(f"WARNING: Requested {pic_num} views, using all {total_available} available")
            pic_num = total_available
        selected_indices = random.sample(range(total_available), pic_num)
        print(f"Selected indices: {selected_indices}")
        cam_matrices = [cam_matrices[i] for i in selected_indices]
        image_paths = [image_paths[i] for i in selected_indices]

    rotated_paths = []
    depth_paths   = []
    for idx, img_path in enumerate(image_paths):
        im = Image.open(img_path)
        print("image path",img_path)
        fname = f"{idx:04d}_color.png"
        out_path = os.path.join(output_dir, fname)
        im.save(out_path)
        rotated_paths.append(out_path)

        depth_src   = img_path.replace("_color.png", "_depth.png")
        depth_im    = Image.open(depth_src)
        depth_fname = f"{idx:04d}_depth.png"
        depth_out   = os.path.join(output_dir, depth_fname)
        depth_im.save(depth_out)
        depth_paths.append(depth_out)

    return meshes, rotated_paths, cam_matrices, depth_paths

def matrix_to_list(m):
    return [list(row) for row in m]

def get_scene_bounding_box(room):
    bbox_array = np.array(room.get_bound_box())
    bb_min = bbox_array.min(axis=0)
    bb_max = bbox_array.max(axis=0)
    print("bb_min:", bb_min)
    print("bb_max:", bb_max)
    return bb_min, bb_max

def visualize_candidate_points(candidate_points, sphere_radius=0.1, color=(1.0, 0.0, 0.0, 1.0)):
    debug_objs = []
    debug_mat = bproc.material.create("debug_material")
    debug_mat.set_principled_shader_value("Base Color", color)
    debug_mat.set_principled_shader_value("Emission Strength", 5.0)
    
    for pt in candidate_points:
        debug_sphere = bproc.object.create_primitive("SPHERE", radius=sphere_radius)
        debug_sphere.set_location(pt)
        debug_sphere.replace_materials(debug_mat)
        debug_objs.append(debug_sphere)
    
    return debug_objs

def create(room_name):  
    # global location_list
    bproc.init()
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.gravity = (0.0, +9.81, 0.0)                             # Set gravity to +Y
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)
    mesh, images, cam_matrices, depth_paths = load_ikea(args.scene, room_name, output_dir, use_smooth_shading=True, pic_num=pic_num)
    print(f"depth_paths: {depth_paths}")

    print("mesh : ", mesh)
    print("mesh0 : ",mesh[0])
    room = mesh[0]
    size = room.get_bound_box()
    size_x = max(abs(size[:, 0]))
    size_y = max(abs(size[:, 1]))
    size_z = max(abs(size[:, 2]))
    print(size_x, size_y, size_z)
    room.set_rotation_euler([math.radians(0), 0, 0])                           # Rotate the scene by -90 degrees along the x-axis
    bpy.context.view_layer.objects.active = room.blender_obj                   # Required for IKEA dataset
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False) #

    room.set_scale([roomscale, roomscale, roomscale])
    room.enable_rigidbody(active=False, collision_shape="MESH")
    bpy.context.object.is_shadow_catcher = True                                 # Set the room as a shadow catcher

    setShadows = args.setShadows 
    if not setShadows:
        room.blender_obj.hide_render = True

    model_objects = args.objects
    model_num = random.choice(list(range(5,7)))
    models = []
    cates = os.listdir(os.path.join(model_objects, split))
    print("Objects :", cates)
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
        json_path = obj_path.replace('.obj', '.json')
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        obj_scale = json_data["scale_factor"]
        obj_scale = obj_scale * 1000 # Convert to mm
        print(f"Loaded: {obj_path}, scale_factor: {obj_scale}")
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
    '''点光源的设置 - Встановлення точкових джерел світла :)'''
    lights = []
    num_lights = 5
    light_class = 'POINT'
    center_init = np.mean(np.array([room_dict[room_name][0], room_dict[room_name][1]]), axis = 0)
    center_init[1] = room_dict[room_name][0][1]
    center_init_scaled = np.array(center_init) * roomscale
    for i in range(num_lights):
        light = bproc.types.Light()
        light.set_type(light_class)
        light.set_energy(random.uniform(100, 250) * roomscale)
        light_location = bproc.sampler.shell(
            center=center_init_scaled,
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
        obj_scale = float(obj_scale)
        size = obj.get_bound_box()
        size_x = max(abs(size[:, 0]))
        size_y = max(abs(size[:, 1]))
        size_z = max(abs(size[:, 2]))
        if obj_scale > max_size:
            max_size = obj_scale
        real_size = [size_x, size_y, size_z, obj_scale * roomscale * 0.001] #0.001 to come back in meter
        sizes.append(real_size)
        # 前者表示归一化模型的scale，后者表示模型从[0,1]^3单位空间中的放缩
        # obj.set_scale([0.2, 0.2, 0.2])
        # obj.set_scale([roomscale, roomscale, roomscale])
        obj.set_scale([obj_scale * roomscale * 0.001, obj_scale * roomscale * 0.001, obj_scale * roomscale * 0.001])
        # # print(obj.get_bound_box())
        obj.enable_rigidbody(active=True)
    print('max_size:', max_size)
    print('sizes:', sizes)
    intrinsics = [
        [577.5, 0.0,   319.5],
        [0.0,   577.5, 239.5],
        [0.0,   0.0,     1.0]
    ]
    bproc.camera.set_intrinsics_from_K_matrix(intrinsics, 640, 480)

    # New objects placement with camera rays
    def sample_objects_via_camera_rays(objs, cam2world, intrinsics, num_candidates=1000, threshold=0.95):
        candidates = []
        cam_location = cam2world.to_translation()
        cam_rot = cam2world.to_3x3()
        
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        cx = intrinsics[0][2]
        cy = intrinsics[1][2]
        
        image_width = 640
        image_height = 480
        
        for i in range(num_candidates):
            u = random.uniform(0, image_width)
            v = random.uniform(0, image_height)
            dir_cam = Vector(((u - cx) / fx, (v - cy) / fy, 1.0))
            dir_cam.normalize()
            direction = cam_rot @ dir_cam
            direction.normalize()
            hit, hit_location, hit_normal, _, hit_obj, _ = bproc.object.scene_ray_cast(cam_location, direction)
            if hit:
                hit_normal_vec = Vector(hit_normal)
                dot_val = hit_normal_vec.dot(Vector((0, 1, 0)))
                # print(f"Ray {i}: hit_location = {hit_location}, dot = {dot_val:.3f}")
                if abs(dot_val) >= threshold:
                    print(f"Ray {i} is candidate, hit_location = {hit_location}, dot = {dot_val:.3f}")
                    candidates.append(Vector(hit_location))
        
        print(f"Found {len(candidates)} candidate points from {num_candidates} launched rays")
        return candidates

    def iterative_object_placement(objs, room, cam2world_matrix, intrinsics, candidate_points, vertical_offset_value=0.2, max_attempts=10):
        bb_min, bb_max = get_scene_bounding_box(room)
        pending_objects = list(objs)
        vertical_offset = Vector((0, -vertical_offset_value, 0))
        attempt = 0
        while pending_objects and attempt < max_attempts:
            attempt += 1
            print(f"Placement attempt {attempt}: {len(pending_objects)} objects pending.")
            for obj in pending_objects:
                chosen_point = random.choice(candidate_points)
                initial_position = Vector(chosen_point) + vertical_offset
                obj.set_location(initial_position)
                obj.set_rotation_euler(bproc.sampler.uniformSO3())
            bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=0.1, max_simulation_time=5, check_object_interval=1, substeps_per_frame=1)
            still_pending = []
            for obj in pending_objects:
                loc = obj.get_location()
                if (loc[0] >= bb_min[0] and loc[0] <= bb_max[0] and
                    loc[1] >= bb_min[1] and loc[1] <= bb_max[1] and
                    loc[2] >= bb_min[2] and loc[2] <= bb_max[2]):
                    print(f"Object {obj.get_cp('instance_id')} placed successfully: {loc}")
                else:
                    print(f"Object {obj.get_cp('instance_id')} out of bounds: {loc}")
                    still_pending.append(obj)
            pending_objects = still_pending
            if pending_objects:
                print(f"After attempt {attempt}, {len(pending_objects)} objects remain pending.")
        if not pending_objects:
            print("All objects placed successfully within scene bounds.")
        else:
            print("Failed to place all objects within maximum attempts.")
        return objs
    
    roll_180 = Matrix.Rotation(math.pi, 4, 'Z')
    for matrix in cam_matrices:
        scaled_matrix = matrix * roomscale
        cam2world_blender = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
            scaled_matrix, ["-X", "Y", "-Z"]
        )
        cam2world = cam2world_blender @ roll_180
        bproc.camera.add_camera_pose(cam2world)
    print(f"Added {len(cam_matrices)} camera poses to the scene")
    
    first_cam2world = cam_matrices[0] * roomscale
    candidate_points = sample_objects_via_camera_rays(objs, first_cam2world, intrinsics, num_candidates=1000, threshold=0.95)
    final_objs = iterative_object_placement(objs, room, first_cam2world, intrinsics, candidate_points, vertical_offset_value=0.3, max_attempts=10)

    # Part of the Compositor - necessary to add rgb images as a background
    scene = bpy.context.scene
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    nodes.clear()

    rlayers = nodes.new("CompositorNodeRLayers")
    bg_node = nodes.new("CompositorNodeImage")
    alpha_over = nodes.new("CompositorNodeAlphaOver")
    comp_node = nodes.new("CompositorNodeComposite")

    rlayers.location = (0, 200)
    bg_node.location = (0, -100)
    alpha_over.location = (200, 50)
    comp_node.location = (400, 50)

    seq_image = bpy.data.images.load(images[0])
    seq_image.source = 'SEQUENCE'
    bg_node.image = seq_image

    bg_node.frame_start = 1
    bg_node.frame_duration = len(images)
    bg_node.use_auto_refresh = True

    links.new(bg_node.outputs["Image"], alpha_over.inputs[1])
    links.new(rlayers.outputs["Image"], alpha_over.inputs[2])
    links.new(alpha_over.outputs["Image"], comp_node.inputs["Image"])
    print("Debug of camera poses and backgrounds")
    print(f"Number of cameras: {len(cam_matrices)}")
    print(f"Number of backgrounds: {len(images)}")
    for i, (cam, img_path) in enumerate(zip(cam_matrices, images), start=1):
        print(f"Frame {i} — Camera pose added — Background: {img_path}")
    print("scene.frame_start =", bpy.context.scene.frame_start)
    print("scene.frame_end   =", bpy.context.scene.frame_end)


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
        location = obj.get_location()  # mathutils.Vector
        rotation_mat = obj.get_rotation_mat()  # mathutils.Matrix
        scale = obj.get_scale()
        local2world_mat = obj.get_local2world_mat()  # mathutils.Matrix

        category_ids.append(category_id)
        instance_ids.append(instance_id)
        locations.append(location) #list(location))
        rotation_mats.append(matrix_to_list(rotation_mat))
        scales.append(scale)
        local2world_mats.append(matrix_to_list(local2world_mat))
        
        if location[2] < room_dict[room_name][0][2] - 20:
            error_num += 1

    if error_num >= 3 and os.path.exists(output_dir):
        os.rmdir(output_dir)
        print(output_dir)
        return

    gt_data = dict()
    gt_data["category_id"] = category_ids
    gt_data["instance_id"] = instance_ids
    gt_data["location"] = locations
    gt_data["rotation_mat"] = rotation_mats
    gt_data["scale"] = scales
    gt_data["size"] = sizes
    gt_data["local2world_mat"] = local2world_mats
    gt_data["cam2world_matrix"] = [matrix_to_list(m) for m in cam_matrices]

    gt_file = os.path.join(output_dir, 'gt.pkl')
    with open(gt_file, "wb") as f:
        pickle.dump(gt_data, f)
    f.close()

    bproc.renderer.enable_segmentation_output(map_by= "instance_id", default_values={'instance_id': 255})
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    data = bproc.renderer.render()
    nocs_data = bproc.renderer.render_nocs()
    data.update(nocs_data)

    special_scenes = ['table17', 'table29']
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
                    ikea_path = depth_paths[id]
                    depth_ikea = cv2.imread(ikea_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    depth_ikea *= roomscale*10
                    if room_name in special_scenes:
                        far_plane = array >= 1e10
                        array = np.where(far_plane, depth_ikea, array)
                    else:
                        seg = data['instance_id_segmaps'][id]
                        mask_obj = (seg > 0) & (seg < 255)
                        array = np.where(mask_obj, array, depth_ikea)
                    print('far:', np.max(array))
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
                    output_file = os.path.join(output_dir, str(id).zfill(4)+'_depth_norm.png')
                    depth_norm = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                    cv2.imwrite(output_file, depth_color)
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
