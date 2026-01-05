from imageio.v2 import imwrite
from skimage.filters import median, threshold_otsu
from skimage.morphology import ball

from image_utils import image_load, detection_edge

def main():
    img = image_load("input.jpg")
    img_clean = median(img, ball(3))
    edges = detection_edge(img_clean)

    t = threshold_otsu(edges)
    edges_bin = edges > t

    imwrite("edges.png", edges_bin.astype("uint8") * 255)

if __name__ == "__main__":
    main()
