import numpy as np
from imageio.v2 import imwrite
from skimage.filters import median, threshold_otsu
from skimage.morphology import ball

from image_utils import image_load, detection_edge

def main():
    # Load image (put your image path here)
    img = image_load("input.jpg")

    # Denoise
    clean_img = median(img, ball(3))

    # Edge detection
    edge_mag = detection_edge(clean_img)

    # Binarization
    t = threshold_otsu(edge_mag)
    edge_bin = edge_mag > t

    # Save result
    out = (edge_bin.astype(np.uint8) * 255)
    imwrite("edges.png", out)

if __name__ == "__main__":
    main()

