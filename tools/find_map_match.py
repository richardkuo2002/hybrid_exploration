
import cv2
import numpy as np
import glob
import os

def find_best_match():
    # Path to uploaded image
    query_img_path = "/home/oem/.gemini/antigravity/brain/5d9addfb-7b14-4f84-84c3-1d3a1b00cd73/uploaded_image_1765332808196.png"
    # Search in ir2_maps recursively
    maps_dir = "maps/ir2_maps"
    
    if not os.path.exists(query_img_path):
        print(f"Error: Query image not found at {query_img_path}")
        return

    # Load query image
    query_img = cv2.imread(query_img_path)
    if query_img is None:
        print("Error: Failed to load query image")
        return
        
    print(f"Loaded query image: {query_img.shape}")
    
    height, width = query_img.shape[:2]
    
    # Heuristic: the first subplot is usually the top 1/3rd roughly
    crop_h = int(height * 0.35)
    real_map_region = query_img[40:crop_h, 0:width] # Skip title
    
    # Convert to grayscale
    gray_query = cv2.cvtColor(real_map_region, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get the map structure
    # In the screenshot, free space is white, obstacles/background black.
    _, thresh_query = cv2.threshold(gray_query, 127, 255, cv2.THRESH_BINARY)
    
    # Find bounding box of white pixels (the map free space)
    coords = cv2.findNonZero(thresh_query)
    if coords is None:
        print("Could not find map content in query image")
        return
        
    x, y, w, h = cv2.boundingRect(coords)
    map_content = thresh_query[y:y+h, x:x+w]
    
    matches = []
    
    files = sorted(glob.glob(os.path.join(maps_dir, "**/*.png"), recursive=True))
    print(f"Scanning {len(files)} maps in {maps_dir}...")
    
    for i, fpath in enumerate(files):
        # Load candidate
        candidate = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if candidate is None:
            continue
            
        # Resize candidate to match query bounding box
        # Note: Interpolation might introduce noise, so threshold again
        # If the map shapes are different (aspect ratio), resizing forces them to match, 
        # but the content will be distorted and IoU will ideally be lower than correct match.
        try:
            resized_candidate = cv2.resize(candidate, (w, h))
        except Exception as e:
            continue

        _, thresh_candidate = cv2.threshold(resized_candidate, 127, 255, cv2.THRESH_BINARY)
        
        # IoU Score
        f1 = map_content.flatten() > 0
        f2 = thresh_candidate.flatten() > 0
        
        intersection = np.logical_and(f1, f2).sum()
        union = np.logical_or(f1, f2).sum()
        
        score = 0 if union == 0 else intersection / union
        
        matches.append((score, fpath))
        
        if i % 100 == 0:
            print(f"Processed {i}...")

    # Sort matches
    matches.sort(key=lambda x: x[0], reverse=True)
    
    print("\nTop 5 Matches:")
    for score, name in matches[:5]:
        print(f"{name}: {score:.4f}")

if __name__ == "__main__":
    find_best_match()
