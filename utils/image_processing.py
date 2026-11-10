import cv2
import numpy as np

IMG_SIZE = 128
HIST_BINS = 32

def process_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Invalid image file.")

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Color histograms (H/S/V), not raw pixels -- captures the
    # green/yellow/brown ratios that indicate leaf health instead of
    # memorizing background/lighting/leaf position from ~700 photos.
    features = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [HIST_BINS], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)

    return np.concatenate(features)