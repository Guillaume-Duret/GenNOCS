import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import argparse

def load_image_pair(base_path):
    """Load RGB, depth, and canny images for a given base filename"""
    rgb_path = base_path + '.jpg'
    depth_path = base_path + '_depth.png'
    canny_path = base_path + '_canny.png'
    
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    canny = cv2.imread(canny_path, cv2.IMREAD_GRAYSCALE)
    
    if rgb is None or depth is None or canny is None:
        return None, None, None
    
    return rgb, depth, canny

def extract_shape_features(rgb, depth, canny):
    """Extract shape-focused features from the image triplet"""
    features = []
    
    # 1. Depth-based features
    depth_mask = depth > 0  # Assuming background is 0
    depth_roi = depth[depth_mask]
    
    if depth_roi.size > 0:
        # Depth statistics
        features.extend([
            np.mean(depth_roi), 
            np.std(depth_roi),
            np.percentile(depth_roi, 25),
            np.percentile(depth_roi, 75)
        ])
        
        # Depth-based contour features
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, depth_bin = cv2.threshold(depth_norm, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(depth_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Hu Moments (scale/rotation invariant)
            hu_moments = cv2.HuMoments(cv2.moments(largest_contour)).flatten()
            features.extend(hu_moments)
            
            # Convexity defects
            hull = cv2.convexHull(largest_contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(largest_contour, hull)
                if defects is not None:
                    features.append(defects.shape[0])  # Number of convexity defects
                else:
                    features.append(0)
            else:
                features.append(0)
    else:
        features.extend([0] * 8)  # Padding if no depth data
    
    # 2. Canny edge features
    if canny is not None:
        # Edge orientation histogram
        sobelx = cv2.Sobel(canny, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(canny, cv2.CV_32F, 0, 1, ksize=3)
        orientations = np.arctan2(sobely, sobelx) * (180 / np.pi)
        orientations = orientations[canny > 0]
        hist, _ = np.histogram(orientations, bins=18, range=(0, 180))
        features.extend(hist / max(hist.sum(), 1e-6))  # Normalized
        
        # Edge density
        features.append(canny.sum() / (canny.shape[0] * canny.shape[1]))
    
    # 3. RGB-based shape features (ignoring texture)
    if rgb is not None:
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Contour properties
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(cnt)
            features.append(w / max(h, 1))
            
            # Solidity (area / convex hull area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            features.append(cv2.contourArea(cnt) / max(hull_area, 1))
        else:
            features.extend([1, 1])  # Default values
    
    return np.array(features)

def get_base_files(input_folder):
    """Get all base filenames (without extensions)"""
    files = os.listdir(input_folder)
    base_files = set()
    
    for f in files:
        if f.endswith('.jpg'):
            base = f[:-4]
            # Check if all required companion files exist
            if (f'{base}_depth.png' in files and 
                f'{base}_canny.png' in files):
                base_files.add(base)
    
    return sorted(base_files)

def select_diverse_images(input_folder, output_folder, num_images=100):
    """Select the most shape-diverse images"""
    # Get all valid base filenames
    base_files = get_base_files(input_folder)
    if not base_files:
        print("No valid image sets found in the input folder.")
        return
    
    print(f"Found {len(base_files)} image sets. Processing...")
    
    # Extract features for each image set
    features_list = []
    valid_files = []
    
    for base in base_files:
        rgb, depth, canny = load_image_pair(os.path.join(input_folder, base))
        if rgb is None or depth is None or canny is None:
            continue
            
        features = extract_shape_features(rgb, depth, canny)
        features_list.append(features)
        valid_files.append(base)
    
    if not features_list:
        print("No valid images with extractable features found.")
        return
    
    features_array = np.array(features_list)
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_array)
    
    # Use K-Means clustering to select diverse samples
    num_clusters = min(num_images, len(valid_files))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(normalized_features)
    
    # Select the most central image from each cluster
    selected_indices = []
    for i in range(num_clusters):
        cluster_mask = (kmeans.labels_ == i)
        cluster_features = normalized_features[cluster_mask]
        
        if len(cluster_features) == 0:
            continue
            
        # Find the most central image in the cluster
        cluster_center = kmeans.cluster_centers_[i]
        distances = euclidean_distances(cluster_features, [cluster_center])
        closest_idx = np.argmin(distances)
        
        # Get the original index
        original_indices = np.where(cluster_mask)[0]
        selected_indices.append(original_indices[closest_idx])
    
    # If we didn't get enough images, add the most different remaining ones
    if len(selected_indices) < num_images:
        remaining_indices = set(range(len(valid_files))) - set(selected_indices)
        remaining_features = normalized_features[list(remaining_indices)]
        
        if selected_indices:
            selected_features = normalized_features[selected_indices]
            avg_distances = np.mean(euclidean_distances(remaining_features, selected_features), axis=1)
            additional_indices = np.argsort(avg_distances)[-1 * (num_images - len(selected_indices)):]
            
            remaining_indices_list = list(remaining_indices)
            selected_indices.extend([remaining_indices_list[i] for i in additional_indices])
        else:
            selected_indices = np.random.choice(len(valid_files), size=num_images, replace=False).tolist()
    
    # Ensure we have exactly num_images
    selected_indices = selected_indices[:num_images]
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Copy selected DEPTH images to output folder
    for idx in selected_indices:
        base = valid_files[idx]
        src_path = os.path.join(input_folder, f"{base}_depth.png")
        dst_path = os.path.join(output_folder, f"{base}_depth.png")
        
        try:
            depth_img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(dst_path, depth_img)
            print(f"Selected: {base}_depth.png")
        except Exception as e:
            print(f"Failed to copy {base}_depth.png: {e}")
    
    print(f"Selected {len(selected_indices)} diverse depth images to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select shape-diverse depth images from aligned object dataset.')
    parser.add_argument('--input', type=str, required=True, help='Input folder with image sets')
    parser.add_argument('--output', type=str, required=True, help='Output folder for selected depth images')
    parser.add_argument('--num', type=int, default=100, help='Number of diverse images to select')
    
    args = parser.parse_args()
    
    select_diverse_images(args.input, args.output, args.num)
