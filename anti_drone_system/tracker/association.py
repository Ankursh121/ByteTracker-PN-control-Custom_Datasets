import numpy as np
from scipy.optimize import linear_sum_assignment

def iou_distance(atracks, btracks):
    """
    Computes distance based on Intersection over Union (IoU) between two lists of tracks/detections.
    Returns:
        cost_matrix: a 2D numpy array where cost_matrix[i, j] = 1 - IoU(atracks[i], btracks[j])
    """
    if len(atracks) == 0 or len(btracks) == 0:
        return np.empty((len(atracks), len(btracks)), dtype=np.float32)

    cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    
    for i, atrack in enumerate(atracks):
        # Support both Track objects and raw box arrays [x1, y1, x2, y2]
        box_a = atrack.tlbr if hasattr(atrack, 'tlbr') else atrack[:4]
        
        for j, btrack in enumerate(btracks):
            box_b = btrack.tlbr if hasattr(btrack, 'tlbr') else btrack[:4]
            
            # Intersection coordinates
            x1 = max(box_a[0], box_b[0])
            y1 = max(box_a[1], box_b[1])
            x2 = min(box_a[2], box_b[2])
            y2 = min(box_a[3], box_b[3])
            
            if x2 > x1 and y2 > y1:
                intersection_area = (x2 - x1) * (y2 - y1)
                area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                union_area = area_a + area_b - intersection_area
                iou = intersection_area / max(union_area, 1e-6)
            else:
                iou = 0.0
                
            cost_matrix[i, j] = 1.0 - iou
            
    return cost_matrix

def linear_assignment(cost_matrix, thresh):
    """
    Solves the linear assignment problem using the Hungarian algorithm.
    Filters out matches with cost higher than the threshold.
    Returns:
        matches: list of tuples (row_idx, col_idx) representing matched tracks/detections
        unmatched_a: list of row indices with no match
        unmatched_b: list of col indices with no match
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
    matches = []
    # Solve linear sum assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    unmatched_a = []
    unmatched_b = []
    
    # Track which rows/cols are matched
    matched_rows = set()
    matched_cols = set()
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > thresh:
            continue
        matches.append((r, c))
        matched_rows.add(r)
        matched_cols.add(c)
        
    for r in range(cost_matrix.shape[0]):
        if r not in matched_rows:
            unmatched_a.append(r)
            
    for c in range(cost_matrix.shape[1]):
        if c not in matched_cols:
            unmatched_b.append(c)
            
    return np.array(matches, dtype=int), unmatched_a, unmatched_b
