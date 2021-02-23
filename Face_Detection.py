from keras.models import load_model
from PIL import Image
import numpy as np
import mtcnn
import os
import cv2

from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
print(os.getcwd())
import time


def extract_face(filename, required_size=(160, 160)):

    list_of_faces_in_image = []
     # load image from file
    # image = Image.open(filename)
    image = filename
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # convert to RGB, if needed
    # image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    print(type(pixels))

    # pixels_2 = pixels
    pixels_2 = cv2.resize(pixels, (60, 60))
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    bas_time = time.time()
    results = detector.detect_faces(pixels)
    print(time.time()- bas_time)
    for sep in results:
        x1, y1, width, height = sep['box']
        print(x1, y1, width, height)
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        list_of_faces_in_image.append((x1, y1, x2, y2))

        Image_2 = cv2.rectangle(pixels_2, (x1, y1), (x2, y2), (0, 255, 2555), 2)
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
    return list_of_faces_in_image

# print(extract_face(r'C:\Users\Desktop\ayse_proje\examples\faces.jpg'))
