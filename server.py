from fastapi import FastAPI
from engine.FaceApi import FaceApi
from database.EsClient import EsClient
from datetime import datetime

app = FastAPI()
es_client = EsClient("localhost", "9200")
faceapi = FaceApi()


@app.get("/")
def index():
    return {
        "status": 200,
        "msg": "faceapi v.1.0"
    }


@app.post("/register")
def register_face(image):
    res = faceapi.detect_face(image)

    if len(res) == 0:
        return {
            "status": 400,
            "msg": "No face detected"
        }
    else:
        doc = {
            "uuid": str(res[0]),
            "filename": str(res[0]) + ".jpg",
            "embedding": res[1],
            "timestamp": datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        }

        result = es_client.add(doc)

        if result:
            return {
                "status": 200,
                "msg": "face stored successfully",
                "uid": res[0]
            }
        else:
            return {
                "status": 400,
                "msg": "could not store face"
            }
