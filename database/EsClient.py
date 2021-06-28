from elasticsearch import Elasticsearch
from typing import Dict


class EsClient:

    def __init__(self, host, port):
        self.es_client = Elasticsearch([{'host': host, 'port': port}])
        self.index_name = "faceapi_index"

    def create_index(self, mapping: Dict):
        self.es_client.indices.create(self.index_name, ignore=400, body=mapping)
        return True

    def add(self, document):
        self.es_client.index(index=self.index_name, doc_type="_doc", body=document)
        return True
