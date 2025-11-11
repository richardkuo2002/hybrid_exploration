#!/usr/bin/env python3
"""
image_to_sdf_world.py  (v2)
Convert a 2D floor-plan image into a Gazebo (gz-sim) SDF world by extruding
black “wall” pixels into boxes, with aggressive piece-count reduction:

- Optional downsampling
- Morphological closing (bridges tiny gaps)
- Vertical merge of horizontal runs into large rectangles
- Minimum rectangle size filter

Edit CONFIG below, then run:  python3 src/mobile_robot/tools/image_to_sdf_world.py
"""

import os
import numpy as np

# Prefer OpenCV (fast); will fall back to Pillow if needed
try:
    import cv2
except Exception:
    cv2 = None
try:
    from PIL import Image
except Exception:
    Image = None

try:
    import yaml
except Exception:
    yaml = None

# ----------------------- CONFIG (EDIT ME) -----------------------
CONFIG = {
    # Either point to a ROS map YAML (preferred), or set IMAGE_PATH/RESOLUTION/ORIGIN_* manually.
    "MAP_YAML": "",  # e.g. "/home/you/maps/office.yaml" (leave "" to disable)

    # If no MAP_YAML, use these:
    "IMAGE_PATH": os.path.expanduser("/home/mouad/ws_mobile/src/mobile_robot/tools/floor_plan.png"),
    "RESOLUTION": 0.05,     # meters/pixel
    "ORIGIN_X": 0.0,        # world X of image pixel (0,0)
    "ORIGIN_Y": 0.0,        # world Y of image pixel (0,0)
    "FLIP_Y": True,         # True: +Y up in world

    # Geometry of extruded walls
    "WALL_HEIGHT": 2.5,     # meters
    "WALL_THICKNESS": 0.10, # meters (thickness along Y for horizontal strips, but we merge vertically now)

    # Binarization
    "THRESHOLD": 128,       # pixel < THRESHOLD => wall

    # --- Piece-count reduction knobs ---
    "DOWNSAMPLE_FACTOR": 1,     # 1 = off; 2 halves both dims (area/4); 3,4...
    "MORPH_CLOSE_SIZE": 3,      # 0=off; 3 or 5 is good to bridge 1px gaps
    "MERGE_TOL_PX": 1,          # allow x-run to shift by this many px across rows and still be merged
    "MIN_RECT_SIZE_M": 0.15,    # drop rectangles smaller than this on either side (meters)

    # Output
    "OUTPUT_SDF": os.path.expanduser("~/ws_mobile/src/mobile_robot/model/floorplan.world"),
}
# ---------------------------------------------------------------

SDF_HEADER = """<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="floorplan_world">
    <gravity>0 0 -9.8</gravity>
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
    </scene>
    <!-- Ground plane: pure white (no textures) -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material>
            <diffuse>1 1 1 1</diffuse>
            <ambient>1 1 1 1</ambient>
            <specular>0 0 0 1</specular>
          </material>
        </visual>
      </link>
    </model>
"""

SDF_FOOTER = """
  </world>
</sdf>
"""

def read_yaml_map(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    img_path = data["image"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(path), img_path)
    resolution = float(data["resolution"])
    origin = data.get("origin", [0, 0, 0])
    origin_x, origin_y = float(origin[0]), float(origin[1])
    negate = int(data.get("negate", 0))
    occ_th = float(data.get("occupied_thresh", 0.65))
    # For negate==0, occupied if pixel < 255*(1-occ_th)
    if negate == 0:
        threshold = int(round(255 * (1.0 - occ_th)))
    else:
        threshold = int(round(255 * occ_th))
    return {
        "image": img_path,
        "resolution": resolution,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "threshold": threshold,
        "negate": negate,
    }

def load_gray(path):
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img
    if Image is not None:
        return np.array(Image.open(path).convert("L"))
    raise RuntimeError("Need OpenCV (python3-opencv) or Pillow (python3-pil) to read images.")

def add_box_model(name, cx, cy, yaw, sx, sy, sz, color=(0.2,0.2,0.2,1.0)):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{cx:.4f} {cy:.4f} {sz/2:.4f} 0 0 {yaw:.6f}</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material><ambient>{color[0]} {color[1]} {color[2]} {color[3]}</ambient></material>
        </visual>
      </link>
    </model>
"""

def binarize(img, threshold, negate_from_yaml=0):
    if negate_from_yaml == 1:
        img = 255 - img
    walls = (img < int(threshold)).astype(np.uint8)
    return walls

def downsample(img_u8, factor):
    if factor <= 1:
        return img_u8
    if cv2 is not None:
        h, w = img_u8.shape
        return cv2.resize(img_u8, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    # Pillow fallback
    pil = Image.fromarray(img_u8)
    w, h = pil.size
    pil_small = pil.resize((w // factor, h // factor), resample=Image.BOX)
    return np.array(pil_small)

def morph_close(img_u8, ksize):
    if ksize <= 0 or cv2 is None:
        return img_u8
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(img_u8, cv2.MORPH_CLOSE, kernel)

def runs_in_row(row):
    """Return list of (x0,x1) for contiguous 1s in a binary row (x1 exclusive)."""
    runs = []
    x = 0
    W = row.shape[0]
    while x < W:
        if row[x] == 0:
            x += 1
            continue
        x0 = x
        while x < W and row[x] == 1:
            x += 1
        x1 = x
        runs.append((x0, x1))
    return runs

def merge_rows_to_rects(walls, merge_tol_px=1):
    """
    Take a binary image (1=wall) and turn it into big axis-aligned rectangles
    by merging horizontally contiguous runs across consecutive rows.

    Returns list of rectangles in pixel coords: (x0,x1,y0,y1) with x1,y1 exclusive.
    """
    H, W = walls.shape
    active = []  # list of dicts: {"x0":..,"x1":..,"y0":..,"y1":..}
    rects = []

    for y in range(H):
        row_runs = runs_in_row(walls[y, :])
        matched_flags = [False] * len(active)
        new_active = []

        for (rx0, rx1) in row_runs:
            # try match with an active rect (same-ish x span)
            best = None
            for i, rect in enumerate(active):
                # allow small drift in x0/x1
                if (abs(rect["x0"] - rx0) <= merge_tol_px) and (abs(rect["x1"] - rx1) <= merge_tol_px):
                    best = i
                    break
            if best is not None:
                # extend that rect one row down
                rect = active[best]
                rect["y1"] = y + 1
                rect["x0"] = min(rect["x0"], rx0)
                rect["x1"] = max(rect["x1"], rx1)
                matched_flags[best] = True
                new_active.append(rect)
            else:
                # start a new rect at this row
                new_active.append({"x0": rx0, "x1": rx1, "y0": y, "y1": y + 1})

        # any unmatched actives are done; flush them to rects
        for rect, matched in zip(active, matched_flags):
            if not matched:
                rects.append(rect)

        active = new_active

    # flush remaining
    rects.extend(active)
    return rects

def main():
    cfg = CONFIG.copy()

    # Pull params from YAML if provided
    negate_from_yaml = 0
    if cfg["MAP_YAML"] and yaml is not None and os.path.exists(cfg["MAP_YAML"]):
        y = read_yaml_map(cfg["MAP_YAML"])
        cfg["IMAGE_PATH"] = y["image"]
        cfg["RESOLUTION"] = y["resolution"]
        cfg["ORIGIN_X"] = y["origin_x"]
        cfg["ORIGIN_Y"] = y["origin_y"]
        cfg["THRESHOLD"] = y["threshold"]
        negate_from_yaml = y["negate"]

    # Load gray image
    img = load_gray(cfg["IMAGE_PATH"])

    # Optional downsample (before thresholding)
    if cfg["DOWNSAMPLE_FACTOR"] > 1:
        img = downsample(img, cfg["DOWNSAMPLE_FACTOR"])

    # Binarize (1 = wall)
    walls = binarize(img, cfg["THRESHOLD"], negate_from_yaml)

    # Optional morphological closing to bridge tiny gaps
    walls = morph_close(walls, cfg["MORPH_CLOSE_SIZE"])

    H, W = walls.shape
    # Merge into rectangles
    rects_px = merge_rows_to_rects(
        walls,
        merge_tol_px=int(cfg["MERGE_TOL_PX"])
    )

    # Convert pixel rects to SDF box models
    res = float(cfg["RESOLUTION"])
    ox, oy = float(cfg["ORIGIN_X"]), float(cfg["ORIGIN_Y"])
    flip_y = bool(cfg["FLIP_Y"])
    height = float(cfg["WALL_HEIGHT"])
    min_m = float(cfg["MIN_RECT_SIZE_M"])

    # Write SDF
    os.makedirs(os.path.dirname(cfg["OUTPUT_SDF"]), exist_ok=True)
    with open(cfg["OUTPUT_SDF"], "w") as f:
        f.write(SDF_HEADER)

        kept = 0
        dropped = 0
        for r in rects_px:
            x0, x1, y0, y1 = r["x0"], r["x1"], r["y0"], r["y1"]
            # size in meters (axis-aligned rectangle)
            sx = (x1 - x0) * res
            sy = (y1 - y0) * res
            # drop tiny pieces
            if sx < min_m or sy < min_m:
                dropped += 1
                continue
            # center in meters (pixel centers at i+0.5)
            cx_pix = (x0 + x1) / 2.0
            cy_pix = (y0 + y1) / 2.0
            cx = ox + cx_pix * res
            cy = oy + (-cy_pix * res if flip_y else cy_pix * res)
            # Plain dark gray walls (no textures)
            name = f"wall_{y0}_{x0}"
            f.write(add_box_model(name, cx, cy, 0.0, sx, sy, height, color=(0.2,0.2,0.2,1.0)))
            kept += 1

        f.write(SDF_FOOTER)

    print(f"[image_to_sdf_world] wrote: {cfg['OUTPUT_SDF']}")
    print(f"  image: {cfg['IMAGE_PATH']}  size(px): {W}x{H}")
    print(f"  res: {cfg['RESOLUTION']} m/px   origin: ({cfg['ORIGIN_X']}, {cfg['ORIGIN_Y']})  flipY: {cfg['FLIP_Y']}")
    print(f"  downsample: {cfg['DOWNSAMPLE_FACTOR']}  morph_close: {cfg['MORPH_CLOSE_SIZE']}  merge_tol_px: {cfg['MERGE_TOL_PX']}")
    print(f"  rectangles kept: {kept}, dropped (min size): {dropped}")

if __name__ == "__main__":
    main()
