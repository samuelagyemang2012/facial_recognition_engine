from elasticsearch import Elasticsearch
from typing import Dict
import json


class EsClient:

    def __init__(self, host, port):
        self.es_client = Elasticsearch([{'host': host, 'port': port}])

    def create_index(self, index_name, mapping: Dict):
        try:
            self.es_client.indices.create(index_name, ignore=400, body=mapping)
            return True
        except:
            return False

    def add(self, index_name, document):
        try:
            self.es_client.index(index=index_name, doc_type="_doc", body=document)
            return True
        except:
            return False

    def search(self, index_name, query):

        # try:
        data = self.es_client.search(index=index_name, body=query)
        return data
    # except:
    #     return False
