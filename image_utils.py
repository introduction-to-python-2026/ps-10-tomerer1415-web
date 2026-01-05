import numpy as np
from PIL import Image
from scipy.ndimage import convolve

def image_load(path: str) -> np.ndarray:
    # טוען תמונה ומחזיר numpy array
    img = np.array(Image.open(path))
    return img

def detection_edge(image: np.ndarray) -> np.ndarray:
    # 1) המרה לאפור: ממוצע 3 ערוצים
    gray = image.mean(axis=2).astype(np.float64)

    # 2) פילטרים (Prewitt)
    fy = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float64)

    fx = np.array([[-1,  0,  1],
                   [-1,  0,  1],
                   [-1,  0,  1]], dtype=np.float64)

    # 3) קונבולוציה עם padding=0 (constant עם cval=0) וגודל פלט זהה לקלט
    edgeY = convolve(gray, fy, mode="constant", cval=0.0)
    edgeX = convolve(gray, fx, mode="constant", cval=0.0)

    # 4) edgeMAG = edgeX^2 + edgeY^2
    edgeMAG = edgeX**2 + edgeY**2
    return edgeMAG
