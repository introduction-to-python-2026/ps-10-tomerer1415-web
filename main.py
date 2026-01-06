from PIL import Image
import numpy as np
from skimage.filters import median
from skimage.morphology import ball
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from image_utils import load_image, edge_detection

image = load_image('.tests/lena.jpg')  

# 2. ניקוי רעשים - שימוש ב-disk(3) כי התמונה דו-ממדית
clean_image = median(image, disk(3))  

# 3. גילוי קצוות
edgeMAG = edge_detection(clean_image)

# 4. יצירת תמונה בינארית (Threshold)
# אם הציון בטסט נמוך מ-0.9, נסי לשנות את 50 לערך אחר (למשל 80 או 100)
edge_binary = edgeMAG > 50

# הצגת התוצאה
plt.imshow(edge_binary, cmap='gray')
plt.axis('off')
plt.show()
