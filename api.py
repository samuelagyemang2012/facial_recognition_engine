from fastapi import FastAPI
from engine.FaceApi import FaceApi
from database.EsClient import EsClient
from datetime import datetime

INDEX_NAME = "faceapi_index"

app = FastAPI()
es_client = EsClient("localhost", "9200")
faceapi = FaceApi()
print("facenet loaded successfully")


@app.get("/api/")
def index():
    """
    API index

    """
    return {
        "status": 200,
        "message": "faceapi v.1.0"
    }


@app.post("/api/detect_face")
def register_face(image):
    res = faceapi.detect_face(image, True)

    if len(res) == 0:
        return {
            "status": 400,
            "message": "No face detected"
        }
    else:
        doc = {
            "uuid": str(res[0]),
            "filename": str(res[0]) + ".jpg",
            "embedding": res[1],
            "timestamp": datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        }

        result = es_client.add(INDEX_NAME, doc)

        if result:
            return {
                "status": 200,
                "message": "face stored successfully",
                "uid": res[0]
            }
        else:
            return {
                "status": 400,
                "message": "could not store face"
            }


@app.post("/api/recognize_face")
def recogize_face(image):
    pass
    # results = []
    # res = faceapi.detect_face(image, False)
    # time_taken = res['took']
    #
    # for hit in res['hits']['hits']:
    #     if faceapi.distance_metric == faceapi.distance_metrics[0]:
    #         cosim = round(hit["_score"] - 1, 4)
    #         uuid = hit["_source"]["uuid"]
    #         results.append([cosim, uuid])
