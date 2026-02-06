"""
Connector for MongoDB.
"""
import pandas as pd
from pymongo import MongoClient

def load_mongodb(uri, db_name, collection_name, query=None):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    data = list(collection.find(query or {}))
    client.close()
    return pd.DataFrame(data)
