import cv2
import numpy as np

IMG_SIZE = 128

def process_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Invalid image file.")

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = image.flatten()

    return image