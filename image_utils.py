from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(file_path):
    image = Image.open(file_path)
    return np.array(image)

def edge_detection(image):
    if image.ndim == 3:
        # חישוב ממוצע הערוצים להפיכה לאפור
        gray_image = np.mean(image, axis=2)
    else:
        gray_image = image

    # הגדרת קרנלים של Sobel
    kernelX = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    kernelY = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    
    # ביצוע קונבולוציה (שימוש בייבוא ישיר מ-scipy)
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill')
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill')
    
    # חישוב עוצמת הקצוות
    edgeMAG = np.sqrt(edgeX*2 + edgeY*2)
    return edgeMAG
