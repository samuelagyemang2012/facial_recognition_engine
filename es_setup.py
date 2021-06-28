from database.EsClient import EsClient
from database.mapping import faceapi_mapping


es = EsClient("localhost", "9200")

res = es.create_index(faceapi_mapping)

if res:
    print("index created")
