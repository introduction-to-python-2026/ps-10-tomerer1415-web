import numpy as np
from imageio.v2 import imread
from scipy.ndimage import convolve


def image_load(path: str) -> np.ndarray:
    """
    Load image from disk and return as numpy array.
    """
    return imread(path)


def detection_edge(image: np.ndarray) -> np.ndarray:
    """
    Edge detection according to the instructions.
    Works for both RGB (H,W,3) and grayscale (H,W).
    Returns edge magnitude map (H,W).
    """

    # grayscale: mean over channels if RGB
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image

    gray = gray.astype(np.float64)

    # filters (Prewitt)
    fy = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float64)

    fx = np.array([[-1,  0,  1],
                   [-1,  0,  1],
                   [-1,  0,  1]], dtype=np.float64)

    # convolution with padding=0 (constant 0)
    edgeY = convolve(gray, fy, mode="constant", cval=0.0)
    edgeX = convolve(gray, fx, mode="constant", cval=0.0)

    # magnitude (no sqrt, as in instructions)
    edgeMAG = edgeX**2 + edgeY**2
    return edgeMAG
