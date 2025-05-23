# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch
from PIL import Image
import os
import argparse
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.text2image import HunyuanDiTPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

def image_to_3d(image_path='assets/demo.png'):
    rembg = BackgroundRemover()
    model_path = 'tencent/Hunyuan3D-2'

    image = Image.open(image_path)
    if image.mode == 'RGB':
        image = rembg(image)

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    mesh = pipeline(image=image, num_inference_steps=30, mc_algo='mc',
                    generator=torch.manual_seed(2025))[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    mesh.export('mesh.glb')

    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
    mesh = pipeline(mesh, image=image)
    mesh.export('texture.glb')
    

def text_to_3d(prompt='a car'):
    rembg = BackgroundRemover()
    t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
    model_path = 'tencent/Hunyuan3D-2'
    i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    image = t2i(prompt)
    image = rembg(image)
    mesh = i23d(image, num_inference_steps=30, mc_algo='mc')[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    mesh.export('t2i_demo.glb')


def image_to_3d_fast(image_path='assets/demo.png'):
    rembg = BackgroundRemover()
    model_path = 'tencent/Hunyuan3D-2'

    image = Image.open(image_path)

    if image.mode == 'RGB':
        image = rembg(image)

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder='hunyuan3d-dit-v2-0-fast',
        variant='fp16'
    )

    mesh = pipeline(image=image, num_inference_steps=30, mc_algo='mc',
                    generator=torch.manual_seed(2025))[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    mesh.export('mesh.glb')


def image_to_3d_fast_folder_old(image_folder='assets/images'):
    rembg = BackgroundRemover()
    model_path = 'tencent/Hunyuan3D-2'

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder='hunyuan3d-dit-v2-0-fast',
        variant='fp16'
    )

    # Iterate over all files in the folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)

            if image.mode == 'RGB':
                image = rembg(image)

            mesh = pipeline(image=image, num_inference_steps=50, mc_algo='mc',
                            generator=torch.manual_seed(2025))[0]
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh)

            # Generate output mesh filename based on input image name
            mesh_filename = os.path.splitext(filename)[0] + '.glb'
            mesh.export(mesh_filename)

def image_to_3d_fast_folder(image_folder, output_folder):
    rembg = BackgroundRemover()
    model_path = 'tencent/Hunyuan3D-2'

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder='hunyuan3d-dit-v2-0'
    )
    
    """
    pipeline_fast = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder='hunyuan3d-dit-v2-0-fast',
        variant='fp16'
    )

    pipeline_turbo = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder='hunyuan3d-dit-v2-0-turbo',
        variant='fp16'
    )
    """
    
    pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(
            model_path)

    #subfolder='hunyuan3d-paint-v2-0')

    """
    pipeline_tex_turbo = Hunyuan3DPaintPipeline.from_pretrained(
            model_path,
            subfolder='hunyuan3d-paint-v2-0-turbo'
    )
    """

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)

            if image.mode == 'RGB':
                image = rembg(image)

            # Generate output mesh filename based on input image name
            base_name = os.path.splitext(filename)[0]

            # Create a subfolder for this mesh
            mesh_subfolder = os.path.join(output_folder, base_name)
            os.makedirs(mesh_subfolder, exist_ok=True)


            mesh = pipeline(image=image, num_inference_steps=50, octree_resolution=512, mc_algo='mc',
                            generator=torch.manual_seed(2025))[0]
            mesh_filename = base_name + '0.obj'
            output_path = os.path.join(mesh_subfolder, mesh_filename)
            mesh.export(output_path)

            mesh = FloaterRemover()(mesh)
            mesh_filename = base_name + '1.obj'
            output_path = os.path.join(mesh_subfolder, mesh_filename)
            mesh.export(output_path)

            mesh = DegenerateFaceRemover()(mesh)
            mesh_filename = base_name + '2.obj'
            output_path = os.path.join(mesh_subfolder, mesh_filename)
            mesh.export(output_path)

            mesh = FaceReducer()(mesh)

            # Generate output mesh filename based on input image name
            # mesh_filename = os.path.splitext(filename)[0] + '.obj'
            # output_path = os.path.join(output_folder, mesh_filename)
            # mesh.export(output_path)
            # print(f"Saved: {output_path}")
            
            # mesh_tex = pipeline_tex(mesh, image=image)
            # mesh_filename_tex = os.path.splitext(filename)[0] + '_textured.obj'
            # output_path_tex = os.path.join(output_folder, mesh_filename_tex)
            # mesh_tex.export(output_path_tex)
            # print(f"Saved: {output_path_tex}")

            # # Generate output mesh filename based on input image name
            # base_name = os.path.splitext(filename)[0]

            # # Create a subfolder for this mesh
            # mesh_subfolder = os.path.join(output_folder, base_name)
            # os.makedirs(mesh_subfolder, exist_ok=True)

            # --- Non-textured mesh ---
            #mesh_filename = base_name + '.obj'
            #output_path = os.path.join(mesh_subfolder, mesh_filename)
            #mesh.export(output_path)
            #print(f"Saved: {output_path}")

            # --- Textured OBJ ---
            mesh_tex = pipeline_tex(mesh, image=image)

            # OBJ with texture (will generate .obj, .mtl, and .png)
            obj_tex_filename = base_name + '_textured.obj'
            output_path_obj_tex = os.path.join(mesh_subfolder, obj_tex_filename)
            # TODO mesh_tex.export(output_path_obj_tex)
            print(f"Saved: {output_path_obj_tex}")

            # --- GLB (GLTF Binary) with texture ---
            glb_filename = base_name + '_textured.glb'
            output_path_glb = os.path.join(mesh_subfolder, glb_filename)

            # Export as GLB (automatically embeds texture)
            mesh_tex.export(
                output_path_glb
            )
            print(f"Saved: {output_path_glb}")

if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser(description="Convert images in a folder to 3D meshes (GLB format).")
    parser.add_argument('--input', type=str, required=True, help="Path to the folder containing input images.")
    parser.add_argument('--output', type=str, required=True, help="Path to the folder where GLB files will be saved.")
    args = parser.parse_args()

    # image_to_3d_fast()
    # image_to_3d()
    image_to_3d_fast_folder(args.input, args.output)
    # text_to_3d()
