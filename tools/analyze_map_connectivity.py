import cv2
import numpy as np
import os

map_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps", "easy_odd", "img_1001.png")

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

# Find start position
start_points = np.where(img == PIXEL_START)
if start_points[0].size > 0:
    start_y, start_x = start_points[0][0], start_points[1][0]
    print(f"Start position found at: ({start_x}, {start_y})")
else:
    print("Start position not found!")
    exit(1)

# Create binary free map (free space = 255, occupied = 0)
# In simulation: final_map[map_img_int > MAP_THRESHOLD] = PIXEL_FREE (255)
binary_map = np.zeros_like(img, dtype=np.uint8)
binary_map[img > MAP_THRESHOLD] = 255

# Ensure start pos is free
binary_map[start_y, start_x] = 255

# Connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

print(f"Number of connected components (including background): {num_labels}")

# Find component containing start position
start_label = labels[start_y, start_x]
print(f"Start position is in component label: {start_label}")

total_free_pixels = np.sum(binary_map == 255)
start_component_pixels = stats[start_label, cv2.CC_STAT_AREA]

print(f"Total free pixels: {total_free_pixels}")
print(f"Start component size: {start_component_pixels}")
print(f"Max achievable coverage: {start_component_pixels / total_free_pixels * 100:.2f}%")

# Save labeled map for visualization (optional, but good for debugging)
# Normalize labels to 0-255
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

cv2.imwrite("debug_map_connectivity.png", labeled_img)
print("Saved debug_map_connectivity.png")
