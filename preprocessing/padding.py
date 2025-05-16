import numpy as np

def pad_mri_to_shape(volume, target_shape, symmetric=True):
    """
    Pad a 3D MRI volume with zeros to the target shape.

    Parameters:
        volume (np.ndarray): The input 3D array (e.g., shape (160, 176, 208))
        target_shape (tuple): The desired shape (e.g., (160, 192, 224))
        symmetric (bool): If True, pads evenly on both sides. If False, pads at the end.

    Returns:
        np.ndarray: Zero-padded volume with shape == target_shape
    """
    assert volume.ndim == 3, "Input volume must be 3D"
    assert all(t >= v for t, v in zip(target_shape, volume.shape)), "Target shape must be >= input shape in all dimensions"
    
    pad_width = []
    for i in range(3):
        total_pad = target_shape[i] - volume.shape[i]
        if symmetric:
            before = total_pad // 2
            after = total_pad - before
        else:
            before = 0
            after = total_pad
        pad_width.append((before, after))

    padded_volume = np.pad(volume, pad_width=pad_width, mode='constant', constant_values=0)
    return padded_volume