import pandas as pd
import boto3
from botocore.client import Config
import logging

MINIO_ENDPOINT = "http://minio:9000"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
BUCKET_NAME = "mlops"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_drift(current_stats, baseline_stats, threshold=0.2):
    """Check if data drift occurred"""
    drift = False
    for key in baseline_stats:
        if abs(current_stats[key] - baseline_stats[key]) / baseline_stats[key] > threshold:
            logger.warning(f"Drift detected in {key}: {current_stats[key]} vs {baseline_stats[key]}")
            drift = True
    return drift

def compute_stats(df):
    """Compute feature statistics"""
    return {
        'mean_length': df['message'].str.len().mean(),
        'mean_words': df['message'].str.split().str.len().mean(),
        'has_url_ratio': (df['message'].str.contains(r'http', regex=True).sum() / len(df))
    }

def monitor_and_retrain():
    s3 = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version='s3v4')
    )
    
    # Load baseline stats (from training data)
    s3.download_file(BUCKET_NAME, 'curated/sms_clean.parquet', '/tmp/baseline.parquet')
    baseline_df = pd.read_parquet('/tmp/baseline.parquet')
    baseline_stats = compute_stats(baseline_df)
    
    logger.info(f"Baseline stats: {baseline_stats}")
    
    # Load production data (new events)
    try:
        s3.download_file(BUCKET_NAME, 'prod/events.parquet', '/tmp/prod_events.parquet')
        prod_df = pd.read_parquet('/tmp/prod_events.parquet')
        prod_stats = compute_stats(prod_df)
        
        logger.info(f"Production stats: {prod_stats}")
        
        # Check for drift
        if check_drift(prod_stats, baseline_stats):
            logger.info("Drift detected! Triggering retrain...")
            # In production, this would trigger an Argo Workflow
            # For now, just log
            return True
    except Exception as e:
        logger.error(f"Error monitoring: {e}")
    
    return False

if __name__ == "__main__":
    monitor_and_retrain()