import numpy as np
import trimesh
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip
import os

# --- Configuration ---
PLY_FILE = 'output/images.ply'

FEATURES_FILE = 'output/images_clip_features.npy'

TEXT_QUERY = "bed"

TOP_K = 5000

output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"📁 Created output directory: {output_folder}")
OUTPUT_VALIDATION_FILE = os.path.join(output_folder, "clip_validation_cloud.ply")


# --- Main Script ---
# Load a pre-trained CLIP model
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

print("Loading point cloud data and features...")
pc = trimesh.load(PLY_FILE)
features = np.load(FEATURES_FILE)

original_points = np.array(pc.vertices)
original_colors = np.array(pc.colors)

print(f"Data loaded. Total points: {len(original_points)}")
print(f"Feature vector shape: {features.shape}")

# 1. Process the text query to get its feature vector
print(f"Processing text query: '{TEXT_QUERY}'")
text = clip.tokenize([TEXT_QUERY]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

query_feature = text_features.cpu().numpy()


# 2. Calculate cosine similarity between your text query feature and all point cloud features
print(f"Calculating similarity for the text query...")
similarities = cosine_similarity(query_feature, features)[0]

# 3. Find the indices of the top K most similar points
similar_indices = np.argsort(similarities)[-TOP_K:]

print(f"Top {TOP_K} similar point indices found.")

# 4. Create a new color array for visualization
# Start with a grayscale version of the original colors to make the highlight pop
validation_colors = np.mean(original_colors[:, :3], axis=1, keepdims=True).astype(np.uint8)
validation_colors = np.tile(validation_colors, (1, 3))
validation_colors = np.hstack([validation_colors, original_colors[:, 3:4]]) # Keep original alpha


# Color the top K similar points green
for idx in similar_indices:
    validation_colors[idx] = [0, 255, 0, 255]

# 5. Save the new point cloud for inspection
print(f"Saving validation point cloud to {OUTPUT_VALIDATION_FILE}")
validation_pc = trimesh.PointCloud(vertices=original_points, colors=validation_colors)
validation_pc.export(OUTPUT_VALIDATION_FILE)

print("✅ Done! Open 'text_query_visualization.ply' in your favorite point cloud viewer.")
print(f"Look for the GREEN points that correspond to your query: '{TEXT_QUERY}'.")