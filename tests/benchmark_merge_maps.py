
import time
import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np
from parameter import PIXEL_FREE, PIXEL_OCCUPIED, PIXEL_UNKNOWN
from utils import merge_maps_jit

def original_merge_maps(maps_to_merge, shape):
    merged_map = np.ones(shape, dtype=np.uint8) * PIXEL_UNKNOWN
    valid_maps = [m for m in maps_to_merge if isinstance(m, np.ndarray)]
    if not valid_maps:
        return merged_map

    any_obs = np.zeros(shape, dtype=bool)
    any_free = np.zeros(shape, dtype=bool)
    for belief in valid_maps:
        any_obs |= belief == PIXEL_OCCUPIED
        any_free |= belief == PIXEL_FREE

    merged_map[any_free] = PIXEL_FREE
    merged_map[any_obs] = PIXEL_OCCUPIED
    return merged_map

def benchmark():
    # Setup
    H, W = 1000, 1000
    shape = (H, W)
    n_maps = 5
    maps = []
    
    np.random.seed(42)
    for _ in range(n_maps):
        m = np.full(shape, PIXEL_UNKNOWN, dtype=np.uint8)
        # Randomly assign free and occupied
        free_mask = np.random.rand(H, W) < 0.3
        obs_mask = np.random.rand(H, W) < 0.05
        m[free_mask] = PIXEL_FREE
        m[obs_mask] = PIXEL_OCCUPIED # Obstacles overwrite free in generation for simplicity
        maps.append(m)

    # Warmup
    original_merge_maps(maps, shape)

    # Measure
    start_time = time.time()
    n_iters = 50
    for _ in range(n_iters):
        original_merge_maps(maps, shape)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_iters
    print(f"Original merge_maps average time: {avg_time:.4f}s over {n_iters} iterations")

    # Warmup JIT
    merged_jit = np.full(shape, PIXEL_UNKNOWN, dtype=np.uint8)
    merge_maps_jit(maps, merged_jit, PIXEL_FREE, PIXEL_OCCUPIED, PIXEL_UNKNOWN)

    # Measure JIT
    start_time = time.time()
    for _ in range(n_iters):
        merged_jit = np.full(shape, PIXEL_UNKNOWN, dtype=np.uint8)
        merge_maps_jit(maps, merged_jit, PIXEL_FREE, PIXEL_OCCUPIED, PIXEL_UNKNOWN)
    end_time = time.time()
    
    avg_time_jit = (end_time - start_time) / n_iters
    print(f"JIT merge_maps average time:      {avg_time_jit:.4f}s over {n_iters} iterations")
    print(f"Speedup: {avg_time / avg_time_jit:.2f}x")

if __name__ == "__main__":
    benchmark()
