import numpy as np
from imageio.v2 import imread
from scipy.signal import convolve2d

def image_load(path: str) -> np.ndarray:
    """
    Loads a color image and returns it as a numpy array.
    """
    img = imread(path)
    return img

def detection_edge(image: np.ndarray) -> np.ndarray:
    """
    Receives a color image array (H,W,3) and returns edge magnitude map (H,W).
    """
    # Convert to grayscale by averaging channels
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image

    gray = gray.astype(np.float64)

    # Derivative filters
    fy = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float64)

    fx = np.array([[-1,  0,  1],
                   [-1,  0,  1],
                   [-1,  0,  1]], dtype=np.float64)

    # Convolution with zero-padding and same output size
    edgeY = convolve2d(gray, fy, mode="same", boundary="fill", fillvalue=0)
    edgeX = convolve2d(gray, fx, mode="same", boundary="fill", fillvalue=0)

    # Edge magnitude (as defined in the instructions)
    edgeMAG = edgeX**2 + edgeY**2
    return edgeMAG
