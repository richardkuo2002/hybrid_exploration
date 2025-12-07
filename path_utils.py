import numpy as np
from typing import List

def get_bresenham_line(p1, p2):
    """Bresenham's Line Algorithm to get all points between p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    
    points = []
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    err = dx - dy
    
    x, y = x1, y1
    
    while True:
        points.append(np.array([x, y]))
        
        if x == x2 and y == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err = err - dy
            x = x + sx
        if e2 < dx:
            err = err + dx
            y = y + sy
            
    return points

def interpolate_path(path: List[np.ndarray]) -> List[np.ndarray]:
    """Interpolate a list of waypoints into a continuous 1-pixel path."""
    if not path or len(path) < 2:
        return path
        
    full_path = []
    for i in range(len(path) - 1):
        segment = get_bresenham_line(path[i], path[i+1])
        # Avoid duplicating the join point, except for the very first segment
        if i > 0:
            segment = segment[1:] # Skip first point as it is same as last point of previous segment
        
        full_path.extend(segment)
        
    return full_path
