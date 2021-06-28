faceapi_mapping = {
    "mappings": {
        "properties": {
            "uid": {"type": "text"},
            "filename": {"type": "text"},
            "embedding": {"type": "dense_vector",
                          "dims": 128
                          },
            "timestamp": {"type": "text"}
        }
    }
}
