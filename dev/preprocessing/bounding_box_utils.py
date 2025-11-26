import numpy as np

def extract_bounding_box(segmentation_mask, robust=True):
    if robust:
        segmentation_mask = (segmentation_mask / segmentation_mask.max()) > 0.05
    nonzero_indices = np.where(segmentation_mask != 0)

    if nonzero_indices[0].size == 0:
        return None
    min_coords = np.min(nonzero_indices, axis=1)
    max_coords = np.max(nonzero_indices, axis=1)
    return tuple(min_coords), tuple(max_coords)

def get_common_bounding_box(initial_bbox, common_shape):
    """
    Calculates a common-sized bounding box centered at the center of the initial
    bounding box.

    Args:
        initial_bbox (tuple or None): A tuple of two tuples representing the
                                     (min_coords, max_coords) of the initial
                                     bounding box (e.g., ((min_z, min_y, min_x),
                                     (max_z, max_y, max_x))). None if no initial
                                     bounding box was found.
        common_shape (np.ndarray or tuple): A 1D array or tuple of length 3
                                           representing the desired shape (size
                                           in each dimension: z, y, x) of the
                                           common bounding box.

    Returns:
        tuple or None: A tuple of two tuples representing the (min_coords,
                       max_coords) of the common-sized bounding box, centered
                       on the initial bounding box's center. Returns None if
                       initial_bbox is None.
    """
    if initial_bbox is None:
        return None

    min_coords_orig, max_coords_orig = initial_bbox
    center_orig = np.array([(min_coords_orig[i] + max_coords_orig[i]) // 2 for i in range(3)])
    common_half_shape = np.array(common_shape) // 2

    min_coords_common = np.maximum(0, center_orig - common_half_shape).astype(int).tolist()
    max_coords_common = (np.array(min_coords_common) + np.array(common_shape)).astype(int).tolist()

    return tuple(min_coords_common), tuple(max_coords_common)

def extract_region_features(segmentation_mask, brain_vol):
    coords = np.argwhere(segmentation_mask)
    if coords.size == 0:
        return {'volume':0,'surface_area':0,'compactness':0,'elongation':0,'flatness':0}
    
    # Volume (normalized)
    vol = segmentation_mask.sum() / brain_vol
    
    # Surface area approximation
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(segmentation_mask)
    surface_area = segmentation_mask.sum() - eroded.sum()
    surface_area = surface_area / brain_vol  # normalize
    
    # Compactness
    compactness = vol**2 / (surface_area**3 + 1e-6)
    
    # PCA for shape
    cov = np.cov(coords, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    elongation = eigvals[0] / (eigvals[2] + 1e-6)
    flatness = eigvals[1] / (eigvals[2] + 1e-6)
    
    return {
        'volume': vol,
        'surface_area': surface_area,
        'compactness': compactness,
        'elongation': elongation,
        'flatness': flatness
    }
