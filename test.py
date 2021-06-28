from engine.FaceApi import FaceApi
import time
import cv2

url = "http://www.wilmabainbridge.com/images/10kfacedatabase2.jpg"
path = "test_data/aj.jpg"
path2 = "test_data/aj2.png"
path3 = "test_data/aj3.jpg"
path4 = "test_data/aj4.jpg"
path5 = "test_data/jlo.jpg"
path6 = "test_data/jlo1.jpg"

faceapi = FaceApi()

# start = time.time()
res = faceapi.detect_face(url)
print(res)
# end = time.time()
#
# distance = faceapi.recognize_face(path5, path6)
# print(distance)
