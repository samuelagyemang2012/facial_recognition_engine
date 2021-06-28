import validators
import requests
import numpy as np
from PIL import Image
import cv2
from deepface.commons import functions
from deepface.basemodels import Facenet
from scipy.spatial import distance
from matplotlib import pyplot as plt
import uuid


class FaceApi:

    def __init__(self):
        self.faces_path = "faces/"
        self.backend = ["retinaface", "opencv"]
        self.model = Facenet.loadModel()
        print("facenet loaded successfully")
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
            face = functions.preprocess_face(img, target_size=self.target_size, detector_backend=self.backend[0])
            embedding = self.model.predict(face)[0]
            uid = uuid.uuid1()
            path = self.faces_path + str(uid) + ".jpg"
            self.save_face(face, path)

            return [uid, embedding]

        except:
            return False

    def save_face(self, face, path):
        face_img = face.reshape(self.target_size[0], self.target_size[0], 3)
        plt.imsave(path, face_img[:, :, ::-1], format="jpg")

    def recognize_face(self, image1, image2):
        embedding1 = self.detect_face(image1)
        embedding2 = self.detect_face(image2)

        dist = 1 - distance.cosine(embedding1, embedding2)
        return dist

    def show_face(self, face):
        face_img = face.reshape(self.target_size[0], self.target_size[0], 3)
        cv2.imshow("test1", face_img)
        cv2.waitKey(0)
