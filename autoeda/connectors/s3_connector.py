"""
Connector for AWS S3.
"""
import pandas as pd
import boto3

def list_s3_objects(bucket, aws_access_key_id, aws_secret_access_key):
    aws_access_key_id = aws_access_key_id.strip()
    aws_secret_access_key = aws_secret_access_key.strip()
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    response = s3.list_objects_v2(Bucket=bucket)
    return [obj['Key'] for obj in response.get('Contents', [])]

def load_s3(bucket, key, aws_access_key_id, aws_secret_access_key):
    aws_access_key_id = aws_access_key_id.strip()
    aws_secret_access_key = aws_secret_access_key.strip()
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'])
