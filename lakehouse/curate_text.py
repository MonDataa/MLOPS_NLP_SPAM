import pandas as pd
import boto3
from botocore.client import Config
import re
import pyarrow as pa
import pyarrow.parquet as pq

# MinIO configuration
MINIO_ENDPOINT = "http://minio:9000"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
BUCKET_NAME = "mlops"

s3 = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version='s3v4')
)

# Download from raw/
s3.download_file(BUCKET_NAME, 'raw/sms_spam.csv', '/tmp/sms_spam.csv')
messages = pd.read_csv('/tmp/sms_spam.csv')

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove non-alphabetic
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

messages['message'] = messages['message'].apply(clean_text)

# Save to curated/
table = pa.Table.from_pandas(messages)
pq.write_table(table, '/tmp/sms_clean.parquet')

s3.upload_file('/tmp/sms_clean.parquet', BUCKET_NAME, 'curated/sms_clean.parquet')

print("Data curated and saved to MinIO curated/")