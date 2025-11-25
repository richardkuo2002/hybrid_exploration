import cv2
import numpy as np
import os

map_path = "maps/easy_odd/img_1001.png"
PIXEL_START = 208
MAP_THRESHOLD = 150

if not os.path.exists(map_path):
    print(f"Map not found: {map_path}")
    exit(1)

# Load map
img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Failed to load map image")
    exit(1)

print(f"Map shape: {img.shape}")

# Create binary map (free space = 255, occupied = 0)
binary_map = np.zeros_like(img, dtype=np.uint8)
binary_map[img > MAP_THRESHOLD] = 255

# Calculate distance transform to find narrowest passages
dist_transform = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)

# Find the maximum distance (radius of largest circle that fits)
max_dist = np.max(dist_transform)
print(f"Max distance (radius): {max_dist}")
print(f"Max corridor width (diameter): {max_dist * 2}")

# Find the minimum non-zero distance (narrowest passage)
# We ignore 0 distance (walls)
min_dist = np.min(dist_transform[dist_transform > 0])
print(f"Min distance (radius): {min_dist}")
print(f"Min corridor width (diameter): {min_dist * 2}")

# Save distance transform for visualization
dist_img = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imwrite("debug_map_width.png", dist_img)
print("Saved debug_map_width.png")
