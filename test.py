from engine.FaceApi import FaceApi
from database.EsClient import EsClient
import time
import json
import cv2

url = "http://www.wilmabainbridge.com/images/10kfacedatabase2.jpg"
path = "test_data/aj.jpg"
path2 = "test_data/aj2.png"
path3 = "test_data/aj3.jpg"
path4 = "test_data/aj4.jpg"
path5 = "test_data/jlo.jpg"
path6 = "test_data/jlo1.jpg"

faceapi = FaceApi()
es = EsClient("localhost", "9200")
results = list()

start = time.time()

res = faceapi.detect_face(url, False)
# print(res[1])
query = {
    "query": {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "cosineSimilarity(params.queryVector, 'embedding') + 1.0",
                # "source": "1 / (1 + l2norm(params.queryVector, 'title_vector'))",  # euclidean distance
                "params": {
                    "queryVector": res[1]
                }
            }
        }
    }
}

res = es.search("faceapi_index", query)
time_taken = res['took']

for hit in res['hits']['hits']:
    cosim = round(hit["_score"] - 1, 4)
    filename = hit["_source"]["uuid"]
    results.append([cosim, filename])

json_data = {
    "time_taken (ms)": time_taken,
    "results": results
}

print(json_data["results"][0][0])

print(json.dumps(json_data))
