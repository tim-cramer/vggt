import numpy as np
import trimesh
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
PLY_FILE = 'bude_cloud.ply'
FEATURES_FILE = 'bude_cloud_dino_features.npy'
OUTPUT_VALIDATION_FILE = 'validation_cloud.ply'

# Index of the point you want to test.
QUERY_POINT_INDEX = 75313

# Number of similar points to find and highlight.
TOP_K = 7000

# --- Main Script ---
print("Loading data...")
pc = trimesh.load(PLY_FILE)
features = np.load(FEATURES_FILE)
print(features.shape)

original_points = np.array(pc.vertices)
original_colors = np.array(pc.colors)

print(f"Data loaded. Total points: {len(original_points)}")

# 1. Select the feature vector for our query point
query_feature = features[QUERY_POINT_INDEX].reshape(1, -1)
query_point = original_points[QUERY_POINT_INDEX]

# 2. Calculate cosine similarity between our query feature and all other features
print(f"Calculating similarity for point {QUERY_POINT_INDEX}...")
similarities = cosine_similarity(query_feature, features)[0]

# 3. Find the indices of the top K most similar points
similar_indices = np.argsort(similarities)[- (TOP_K + 1) :]

print(f"Top {TOP_K} similar point indices found.")

# 4. Create a new color array for visualization
validation_colors = np.mean(original_colors[:, :3], axis=1, keepdims=True).astype(np.uint8)
validation_colors = np.tile(validation_colors, (1, 3))
validation_colors = np.hstack([validation_colors, original_colors[:, 3:4]]) # Keep original alpha

# Color the similar points yellow
for idx in similar_indices:
    validation_colors[idx] = [255, 255, 0, 255] # Yellow

# Color the original query point red
validation_colors[QUERY_POINT_INDEX] = [255, 0, 0, 255] # Red

# 5. Save the new point cloud for inspection
print(f"Saving validation point cloud to {OUTPUT_VALIDATION_FILE}")
validation_pc = trimesh.PointCloud(vertices=original_points, colors=validation_colors)
validation_pc.export(OUTPUT_VALIDATION_FILE)

print("✅ Done! Open 'validation_cloud.ply' in MeshLab or CloudCompare.")
print(f"Look for the RED point (your query) and see if the YELLOW points are on similar surfaces.")