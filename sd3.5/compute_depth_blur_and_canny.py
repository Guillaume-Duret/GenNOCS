import argparse
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import sys
import os

# Add the depth-fm directory to the Python path
sys.path.append(os.path.abspath("depth-fm"))
from depthfm.dfm import DepthFM
from PIL import Image

def process_image(image_path, depthfm_model, device):
    # Load the image
    img = Image.open(image_path).convert('RGB')
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.dirname(image_path)

    # Skip if the image already has "_blur", "_depth", or "_canny" in its name
    if "_blur" in base_name or "_depth" in base_name or "_canny" in base_name:
        print(f"Skipping {base_name} (already processed)")
        return

    # Convert to tensor and apply Canny edge detection
    img_tensor = TF.to_tensor(img)
    img_grey = cv2.cvtColor(img_tensor.numpy().transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    img_grey_8u = (img_grey * 255).astype('uint8')
    img_canny = cv2.Canny(img_grey_8u, 100, 200)
    canny_image_pil = Image.fromarray(img_canny)
    canny_output_path = os.path.join(save_dir, f"{base_name}_canny.png")
    canny_image_pil.save(canny_output_path)
    print(f"Created: {canny_output_path}")

    # Apply Gaussian Blur
    gaussian_blur = transforms.GaussianBlur(kernel_size=51)
    blurred_image = gaussian_blur(img)
    blur_output_path = os.path.join(save_dir, f"{base_name}_blur.png")
    blurred_image.save(blur_output_path)
    print(f"Created: {blur_output_path}")

    # Process with DepthFM model
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)  # Move to GPU (keep as float32)
    c, h, w = img_tensor.shape[1:]
    img_tensor = F.interpolate(img_tensor, (512, 512), mode='bilinear', align_corners=False)
    with torch.no_grad():
        depth_image = depthfm_model(img_tensor, num_steps=2, ensemble_size=4)
    depth_image = F.interpolate(depth_image, (h, w), mode='bilinear', align_corners=False)
    depth_image = depth_image.squeeze(0).squeeze(0).cpu().numpy()
    depth_image_pil = Image.fromarray((depth_image * 255).astype('uint8'))
    depth_output_path = os.path.join(save_dir, f"{base_name}_depth.png")
    depth_image_pil.save(depth_output_path)
    print(f"Created: {depth_output_path}")

def process_folder(folder_path, depthfm_model, device):
    # Get all image files in the folder
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_formats)]

    if not image_files:
        print(f"No images found in the folder: {folder_path}")
        return

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing: {image_file}")
        process_image(image_path, depthfm_model, device)

def main():
    parser = argparse.ArgumentParser(description="Process images in a folder to generate blur, depth, and canny images.")
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the DepthFM model checkpoint')
    args = parser.parse_args()

    # Set device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        raise RuntimeError("CUDA-enabled GPU is required to run this script.")
    print(f"Using device: {device}")

    # Load the DepthFM model and move to GPU
    depthfm_model = DepthFM(ckpt_path=args.ckpt_path).to(device)
    depthfm_model.eval()

    # Process the folder
    process_folder(args.folder_path, depthfm_model, device)

if __name__ == "__main__":
    main()
