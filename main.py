from engine.FaceApi import FaceApi
import time
import cv2

url = "http://www.wilmabainbridge.com/images/10kfacedatabase2.jpg"
path = "data/aj.jpg"

faceapi = FaceApi()

start = time.time()

embedding = faceapi.detect_face(path)

end = time.time()
# embedding = embedding.reshape(16, 8)
# plt.imsave("data/dd.jpg", embedding)

print(embedding)

print(end-start)
