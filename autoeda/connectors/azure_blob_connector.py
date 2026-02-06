"""
Connector for Azure Blob Storage.
"""
import pandas as pd
from azure.storage.blob import BlobServiceClient

def load_azure_blob(connection_string, container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    stream = blob_client.download_blob().readall()
    return pd.read_csv(pd.io.common.BytesIO(stream))
