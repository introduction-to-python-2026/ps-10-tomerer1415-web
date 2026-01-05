from image_utils import image_load, detection_edge

def main():
    img = image_load("input.jpg")
    edges = detection_edge(img)
    print(edges.shape)

if __name__ == "__main__":
    main()
