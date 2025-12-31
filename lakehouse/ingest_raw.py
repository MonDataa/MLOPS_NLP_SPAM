import pandas as pd
import boto3
from botocore.client import Config

# MinIO configuration
MINIO_ENDPOINT = "http://minio:9000"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
BUCKET_NAME = "mlops"

# Create S3 client
s3 = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version='s3v4')
)

# Create bucket if not exists
try:
    s3.create_bucket(Bucket=BUCKET_NAME)
except s3.exceptions.BucketAlreadyExists:
    pass

# Load data
data_path = '../data/spam_nospam.txt'
messages = pd.read_csv(data_path, sep='\t', names=["label", "message"])

# Save to raw/
raw_path = 'raw/sms_spam.csv'
messages.to_csv('/tmp/sms_spam.csv', index=False)

# Upload to MinIO
s3.upload_file('/tmp/sms_spam.csv', BUCKET_NAME, raw_path)

print("Data ingested to MinIO raw/")