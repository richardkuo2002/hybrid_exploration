import numpy as np
import cv2
import os
import sys

# Check dependencies
try:
    import ray
    print("Ray is installed.")
except ImportError:
    print("Ray is NOT installed.")

try:
    import skimage
    print("Scikit-image is installed.")
except ImportError:
    print("Scikit-image is NOT installed.")

# Check map pixels
map_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps", "easy_even", "img_10.png")

if os.path.exists(map_path):
    img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    unique_vals = np.unique(img)
    print(f"Map {map_path} unique values: {unique_vals}")
    if 208 in unique_vals:
        print("Start position (208) FOUND.")
    else:
        print("Start position (208) NOT FOUND.")
else:
    print(f"Map {map_path} does not exist.")
