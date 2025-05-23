import torch
import random
from diffusers import StableDiffusion3Pipeline
import argparse
from pathlib import Path
import os

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float32) #bfloat16)

#pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

negative_prompt = "shadow, shiny, water, transparent, glass, drops, sliced, cut, peeled, half, grappe, piece, ugly, tiling, out of frame, poorly drawn face, extra limbs, body out of frame, blurry, bad anatomy, blurred, artifacts, bad proportions."

def load_prompts_from_file(file_path):
    """Load prompts from a text file."""
    with open(file_path, 'r') as file:
        prompts = file.readlines()
        # Strip newline characters from each line
        prompts = [prompt.strip() for prompt in prompts]
    return prompts

def get_fruit_name(prompt):
    fruit_name = prompt.split("unsliced ")[1].split(" with")[0]
    return fruit_name

def generate_instances(prompt_index, prompt, text_name, output_path):
    os.makedirs(os.path.dirname(f"{output_path}/{text_name}/{prompt_index}"), exist_ok=True) #why doesn't work ? 
    print("create folder ")
    for image_index in range(15):
        seed = random.randint(0, 2147483647)  
        generator = torch.manual_seed(seed)
        image = pipe(prompt,negative_prompt=negative_prompt,num_inference_steps=50,guidance_scale=4,generator=generator).images[0]
        folder_path = f"{output_path}/{text_name}/{prompt_index}"
        os.makedirs(folder_path, exist_ok=True)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"Folder exists: {folder_path}")
        else:
            print(f"Failed to create folder: {folder_path}")
        image.save(f"{output_path}/{text_name}/{prompt_index}/{text_name}_{prompt_index}_{image_index + 1}.jpg")
        print("generated : ", f"{output_path}/{text_name}/{prompt_index}/{text_name}_{prompt_index}_{image_index + 1}.jpg")

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Retrieve prompts based on ID.")
    
    # Define the 'id' argument
    parser.add_argument('id', type=int, help="ID of the prompt (use -1 to go through all).")
    parser.add_argument('prompt_file', type=str, help="path of the prompt file")
    parser.add_argument('output_path', type=str, help="path of the output files and iamges.")
    args = parser.parse_args()

    path = Path(args.prompt_file)
    filename = path.stem
    print(filename) 

    output_path = args.output_path

    prompts = load_prompts_from_file(args.prompt_file)
    prompt_id=args.id    
    # Print the corresponding prompt or all prompts
    if prompt_id == -1: # loop over all prompts
        for i, prompt in enumerate(prompts):
            print(f"Prompt {i}: {prompt}\n")
            generate_instances(i, prompt, filename, output_path)
    elif 0 <= prompt_id < len(prompts):
        print("0")
        print(f"Prompt {prompt_id}: {prompts[prompt_id]}")
        generate_instances(prompt_id, prompts[prompt_id], filename, output_path)
        generate_instances(prompt_id+1, prompts[prompt_id+1], filename, output_path)
        generate_instances(prompt_id+2, prompts[prompt_id+2], filename, output_path)
    else:
        print(f"Invalid prompt ID: {prompt_id}. Please provide a valid ID between 0 and {len(prompts) - 1}.")

if __name__ == "__main__":
    main()











