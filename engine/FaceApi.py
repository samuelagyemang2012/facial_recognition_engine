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
        self.distance_metrics = ["cosine_similarity", "euclidean"]
        self.cosim_threshold = 0.5
        self.euclidean_threshold = 11
        self.distance_metric = self.distance_metrics[0]
        self.backends = ["retinaface", "opencv"]
        self.backend = "opencv"
        self.model = Facenet.loadModel()
        self.target_size = (160, 160)  # self.model.layers[0].input_shape[1:3]

    def set_faces_path(self, path):
        self.faces_path = path

    def set_distance_metric(self, index):
        self.distance_metric = self.distance_metrics[index]

    def set_cosim_threshold(self, threshold):
        self.cosim_threshold = threshold

    def set_euclidean_threshold(self, threshold):
        self.euclidean_threshold = threshold

    def set_backends(self, index):
        self.backend = self.backends[index]

    def detect_face(self, image, save_face):
        img = None
        backend = self.backend

        if validators.url(image):
            img = Image.open(requests.get(image, stream=True).raw)
            img = np.array(img)
            img = img[:, :, ::-1]
        else:
            img = cv2.imread(image)

        try:
            face = functions.preprocess_face(img, target_size=self.target_size, detector_backend=backend)
            embedding = self.model.predict(face)[0]
            uid = uuid.uuid1()
            path = self.faces_path + str(uid) + ".jpg"

            if save_face:
                self.save_face(face, path)

            return [uid, embedding]

        except:
            return False

    def recognize_face(self, image1, image2):
        dist = ""
        embedding1 = self.detect_face(image1, False)
        embedding2 = self.detect_face(image2, False)

        if self.distance_metric == self.distance_metrics[0]:
            dist = 1 - distance.cosine(embedding1, embedding2)
            return dist
        else:
            dist = distance.euclidean(embedding1, embedding2)
            return dist

    def save_face(self, face, path):
        face_img = face.reshape(self.target_size[0], self.target_size[0], 3)
        plt.imsave(path, face_img[:, :, ::-1], format="jpg")

    def show_face(self, face):
        face_img = face.reshape(self.target_size[0], self.target_size[0], 3)
        cv2.imshow("test1", face_img)
        cv2.waitKey(0)
