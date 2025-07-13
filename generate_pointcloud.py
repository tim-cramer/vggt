import os
import torch
import numpy as np
import trimesh
import argparse
import sys
from PIL import Image
import gc

# Add the vggt repository to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def extract_dino_features(image_paths, target_height, target_width, device, batch_size=4):
    """
    Extracts dense DINOv2 feature maps for a list of images in batches to save memory.
    
    Args:
        image_paths (list): List of paths to the input images.
        target_height (int): The height to upsample features to (matching VGGT's output).
        target_width (int): The width to upsample features to.
        device: The torch device ('cuda' or 'cpu').
        batch_size (int): Number of images to process at once.
    
    Returns:
        np.ndarray: A NumPy array of shape (S, H, W, D) containing the feature map for each image.
    """
    print("🦖 Initializing DINOv2 model...")
    # Load the DINOv2 model and its specific image transformations
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dinov2_model.eval()
    
    # Create DINOv2 transforms that resize to the same dimensions as VGGT
    from torchvision import transforms
    dino_transforms = transforms.Compose([
        transforms.Resize((target_height, target_width)),  # Match VGGT's output dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_features = []
    print(f"🦖 Extracting DINOv2 features for {len(image_paths)} images in batches of {batch_size}...")
    print(f"    Target dimensions: {target_height}x{target_width}")
    
    with torch.no_grad():
        # Process images in smaller batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            print(f"    Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
            
            # Load and transform batch
            pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
            transformed_images = torch.stack([dino_transforms(p) for p in pil_images]).to(device)
            print(f"      Transformed tensor shape: {transformed_images.shape}")
            
            # Get patch features from an intermediate block
            features_dict = dinov2_model.forward_features(transformed_images)
            patch_features = features_dict['x_norm_patchtokens']
            print(f"      Patch features shape: {patch_features.shape}")
            
            # Reshape to a 2D feature map
            B, N, D = patch_features.shape
            
            # Calculate patch grid dimensions based on input size
            # DINOv2 uses 14x14 patches, so patch grid is (H//14, W//14)
            H_patch = target_height // 14
            W_patch = target_width // 14
            expected_patches = H_patch * W_patch
            
            print(f"      Expected patch grid: {H_patch}x{W_patch} = {expected_patches} patches, got {N}")
            
            if expected_patches != N:
                # If mismatch, use the actual number of patches and find best grid
                H_patch = W_patch = int(np.sqrt(N))
                if H_patch * W_patch != N:
                    # Handle non-square grids
                    factors = []
                    for j in range(1, int(np.sqrt(N)) + 1):
                        if N % j == 0:
                            factors.append((j, N // j))
                    # Choose the most square-like factorization
                    H_patch, W_patch = min(factors, key=lambda x: abs(x[0] - x[1]))
                print(f"      Using patch grid: {H_patch}x{W_patch}")
            
            # (B, N, D) -> (B, H_patch, W_patch, D) -> (B, D, H_patch, W_patch)
            feature_map_2d = patch_features.reshape(B, H_patch, W_patch, D).permute(0, 3, 1, 2)
            
            # Upsample the feature map to match the target resolution
            upsampled_features = torch.nn.functional.interpolate(
                feature_map_2d,
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            )
            
            # (B, D, H, W) -> (B, H, W, D) for compatibility
            final_features = upsampled_features.permute(0, 2, 3, 1)
            
            # Move to CPU and store
            all_features.append(final_features.cpu())
            
            # Clear GPU memory
            del transformed_images, features_dict, patch_features, feature_map_2d, upsampled_features, final_features
            torch.cuda.empty_cache()
    
    # Clean up DINOv2 model
    del dinov2_model
    torch.cuda.empty_cache()
    gc.collect()
    
    print("✅ DINOv2 feature extraction complete.")
    return torch.cat(all_features, dim=0).numpy()


def aggregate_points_and_features(points, colors, features, voxel_size):
    """Aggregates points, colors, and features that fall into the same voxel."""
    print(f"🧊 Voxelizing and aggregating points with voxel size {voxel_size}...")
    voxel_indices = np.floor(points / voxel_size).astype(int)
    voxel_dict = {}
    for i in range(len(voxel_indices)):
        voxel_key = tuple(voxel_indices[i])
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = {'points': [], 'colors': [], 'features': []}
        voxel_dict[voxel_key]['points'].append(points[i])
        voxel_dict[voxel_key]['colors'].append(colors[i])
        voxel_dict[voxel_key]['features'].append(features[i])
    num_voxels = len(voxel_dict)
    feature_dim = features.shape[1]
    agg_points = np.zeros((num_voxels, 3), dtype=np.float32)
    agg_colors = np.zeros((num_voxels, 3), dtype=np.float32)
    agg_features = np.zeros((num_voxels, feature_dim), dtype=np.float32)
    for i, key in enumerate(voxel_dict.keys()):
        voxel_data = voxel_dict[key]
        agg_points[i] = np.mean(voxel_data['points'], axis=0)
        agg_colors[i] = np.mean(voxel_data['colors'], axis=0)
        agg_features[i] = np.mean(voxel_data['features'], axis=0)
    print(f"✅ Aggregation complete. Original points: {len(points)}, Aggregated points: {num_voxels}")
    return agg_points, agg_colors, agg_features

def save_point_cloud(points, colors, filename):
    """Saves a colored point cloud to a .ply file."""
    if colors.max() <= 1.0:
        colors = colors * 255
    colors_uint8 = colors.astype(np.uint8)
    pc = trimesh.PointCloud(vertices=points, colors=colors_uint8)
    pc.export(filename)
    print(f"✅ Point cloud saved successfully to {filename}")

def run_vggt_and_create_feature_cloud(args):
    """Main pipeline to run VGGT and generate an enhanced, colored point cloud with features."""
    # --- 1. Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"🔄 Initializing and loading VGGT model on {device}...")
    vggt_model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    vggt_model.eval().to(device)

    # --- 2. Load Images for VGGT ---
    image_paths = sorted([os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_paths: raise ValueError(f"No images found in {args.image_folder}")
    print(f"🔄 Loading {len(image_paths)} images for VGGT...")
    images_tensor = load_and_preprocess_images(image_paths).to(device)

    # --- 3. Run VGGT Inference ---
    print("🚀 Running VGGT model inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = vggt_model(images_tensor)

    # --- 4. Process Predictions & Select Point Cloud Source ---
    print("⚙️ Processing VGGT predictions...")
    B, S, C, H, W = predictions["images"].shape
    
    if args.prediction_mode == 'depth':
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))
        extrinsic_np = extrinsic.squeeze(0).cpu().numpy()
        intrinsic_np = intrinsic.squeeze(0).cpu().numpy()
        depth_map_np = predictions["depth"].squeeze(0).cpu().numpy()
        world_points = unproject_depth_map_to_point_map(depth_map_np, extrinsic_np, intrinsic_np)
        confidence = predictions["depth_conf"].squeeze(0).cpu().numpy()
    else: # 'pointmap'
        world_points = predictions["world_points"].squeeze(0).cpu().numpy()
        confidence = predictions["world_points_conf"].squeeze(0).cpu().numpy()

    # --- 5. Clean up VGGT model and predictions to free memory ---
    print("🧹 Cleaning up VGGT model from GPU memory...")
    images_np = predictions["images"].squeeze(0).cpu().numpy()
    del vggt_model, images_tensor, predictions
    torch.cuda.empty_cache()
    gc.collect()

    # --- 6. Extract REAL DINO Features (with batching) ---
    # We pass the original image paths and the target H, W from VGGT
    dino_features = extract_dino_features(image_paths, H, W, device, batch_size=args.dino_batch_size)

    # --- 7. Get Colors and Flatten Data for Filtering ---
    points_flat = world_points.reshape(-1, 3)
    colors_flat = np.transpose(images_np, (0, 2, 3, 1)).reshape(-1, 3)
    features_flat = dino_features.reshape(-1, dino_features.shape[-1])
    confidence_flat = confidence.reshape(-1)

    # --- 8. Apply All Filters ---
    print(f"🔍 Applying filters...")
    keep_mask = np.ones(len(points_flat), dtype=bool)
    if args.conf_percentile > 0:
        conf_threshold_value = np.percentile(confidence_flat, args.conf_percentile)
        conf_mask = confidence_flat >= conf_threshold_value
        keep_mask &= conf_mask
        print(f"    - Confidence filter kept {np.sum(keep_mask)} points.")

    filtered_points = points_flat[keep_mask]
    filtered_colors = colors_flat[keep_mask]
    filtered_features = features_flat[keep_mask]
    print(f"✅ Filtering complete. Final point count: {len(filtered_points)}")

    # --- 9. Optional Voxel Aggregation ---
    if args.voxel_size > 0:
        agg_points, agg_colors, agg_features = aggregate_points_and_features(
            filtered_points, filtered_colors, filtered_features, args.voxel_size
        )
    else:
        agg_points, agg_colors, agg_features = filtered_points, filtered_colors, filtered_features

    # --- 10. Save Outputs ---
    output_features_npy = args.output_ply.replace(".ply", "_dino_features.npy")
    save_point_cloud(agg_points, agg_colors, args.output_ply)
    np.save(output_features_npy, agg_features)
    print(f"✅ DINO features saved successfully to {output_features_npy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a colored point cloud with DINOv2 features using VGGT.")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_ply", type=str, required=True)
    
    parser.add_argument("--prediction_mode", type=str, default="depth", choices=['depth', 'pointmap'])
    parser.add_argument("--conf_percentile", type=float, default=20.0, help="Filter out the bottom N% of points based on confidence (0-100).")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel size for point cloud aggregation. Set to 0 to disable.")
    parser.add_argument("--dino_batch_size", type=int, default=4, help="Batch size for DINOv2 feature extraction to manage memory.")
    
    args = parser.parse_args()
    run_vggt_and_create_feature_cloud(args)