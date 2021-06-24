import validators
import requests
import numpy as np
from PIL import Image
from deepface.commons import functions
from deepface.basemodels import Facenet
import cv2
import time


class FaceApi:

    def __init__(self):
        self.backend = "retinaface"
        self.model = Facenet.loadModel()
        self.target_size = (160, 160)  # self.model.layers[0].input_shape[1:3]

    def detect_face(self, image):
        img = None

        if validators.url(image):
            img = Image.open(requests.get(image, stream=True).raw)
            img = np.array(img)
            img = img[:, :, ::-1]
        else:
            img = cv2.imread(image)

        try:
            face = functions.preprocess_face(img, target_size=self.target_size)  # , detector_backend=self.backend)
            embedding = self.model.predict(face)[0]
            return embedding

            #############################
            # face_img = face.reshape(self.target_size[0], self.target_size[0], 3)
            # cv2.imshow("test1", face_img)
            # cv2.waitKey(0)
            #############################

        except:
            return "No face detected"
