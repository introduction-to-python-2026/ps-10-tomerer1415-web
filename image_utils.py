def detection_edge(image: np.ndarray) -> np.ndarray:
    # אם התמונה צבעונית – הפוך לאפור
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image

    gray = gray.astype(np.float64)

    fy = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float64)

    fx = np.array([[-1,  0,  1],
                   [-1,  0,  1],
                   [-1,  0,  1]], dtype=np.float64)

    edgeY = convolve(gray, fy, mode="constant", cval=0.0)
    edgeX = convolve(gray, fx, mode="constant", cval=0.0)

    edgeMAG = edgeX**2 + edgeY**2
    return edgeMAG
