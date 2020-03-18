import numpy as np
import cv2


def resize(image, target_dim, normalize=False):
    if image.shape[:2] != target_dim:
        inter_method = cv2.INTER_AREA
        if np.prod(image.shape[:2]) > np.prod(target_dim):
            inter_method = cv2.INTER_LINEAR
        
        resized = cv2.resize(image, target_dim, interpolation=inter_method)
    else:
        resized = image

    if len(resized.shape) == 2:
        resized = resized[:,:,None]

    if normalize:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
        return (resized.astype(np.float32) / 255 - mean) / std

    return resized.astype(np.float32)
