from engine.FaceApi import FaceApi
import cv2
from matplotlib import pyplot as plt

url = "http://www.wilmabainbridge.com/images/10kfacedatabase2.jpg"
path = "data/aj.jpg"

faceapi = FaceApi()

embedding = faceapi.detect_face(path)
# embedding = embedding.reshape(16, 8)
# plt.imsave("data/dd.jpg", embedding)

print(embedding)
