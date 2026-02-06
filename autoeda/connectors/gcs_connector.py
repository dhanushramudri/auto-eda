"""
Connector for Google Cloud Storage.
"""
import pandas as pd
from google.cloud import storage

def load_gcs(bucket_name, blob_name, credentials_path):
    client = storage.Client.from_service_account_json(credentials_path)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_bytes()
    return pd.read_csv(pd.io.common.BytesIO(data))
