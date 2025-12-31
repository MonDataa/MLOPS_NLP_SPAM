# MLOPS NLP Spam Detection - Complete Production Architecture

This project implements a production-grade MLOps pipeline for SMS spam detection using LSTM models, deployed on Kubernetes with complete GitOps and monitoring support.

## Architecture Overview

```
Data Ingestion → MinIO → Feature Store (Feast)
     ↓
Data Curation (Delta Lake)
     ↓
Feature Engineering
     ↓
Distributed Training (Ray + MLflow)
     ↓
Model Registry (MLflow)
     ↓
Inference API (Ray Serve)
     ↓
GitOps Deployment (Argo CD)
     ↓
Monitoring (Prometheus/Grafana) + Drift Detection
```

## Project Structure

```
projet_MLOPS_NLP/
├── data/                      # Raw data (empty by default)
├── lakehouse/                 # Data pipeline (ingestion, curation, monitoring)
│   ├── ingest_raw.py         # Load SMS data to MinIO raw/
│   ├── curate_text.py        # Clean text, save to MinIO curated/
│   └── monitor_drift.py      # Detect data drift, trigger retrain
├── feature_repo/             # Feast feature store
│   ├── features.py
│   └── feature_store.yaml
├── training/                 # Model training code
│   └── train_spam.py         # Ray Train + Ray Tune + MLflow
├── serving/                  # Inference API
│   └── serve_spam.py         # Ray Serve endpoint
├── mlflow/                   # MLflow config & models
├── k8s/                      # Kubernetes manifests
│   ├── minio.yaml           # Object storage
│   ├── mlflow.yaml          # Model tracking
│   ├── jobs.yaml            # Ingestion/curation jobs
│   ├── training-job.yaml    # Training job
│   ├── ray-serve.yaml       # Inference deployment
│   ├── monitoring.yaml      # Prometheus/Grafana
│   └── drift-cronjob.yaml   # Drift detection scheduler
├── argocd/                   # GitOps configurations
│   ├── spam-detection-app.yaml
│   └── ray-serve-rollout.yaml
├── requirements.txt          # Python dependencies
└── Dockerfile               # Container image

```

## 1. Prerequisites

- **Kubernetes**: k3s/k0s/MicroK8s (local or on-prem)
- **kubectl**: Configured to access your cluster
- **Git**: Repository for GitOps (GitLab/Gitea/GitHub)
- **Registry**: Harbor or local Docker registry
- **Storage**: Local PV or Longhorn/Rook-Ceph for production

## 2. Data Lake (MinIO + Delta)

### Deployment

```bash
kubectl apply -f k8s/minio.yaml
```

### Bucket Structure

```
s3://mlops/
├── raw/              # Original SMS data
│   └── sms_spam.csv
├── curated/          # Cleaned text (Parquet)
│   └── sms_clean.parquet
└── features/         # Engineered features (Delta/Parquet)
    └── nlp_features_delta/
```

### Features Stored

- `text_length`: Word count
- `word_count`: Number of words
- `has_url`: Contains URL flag
- `has_email`: Contains email flag
- Tokenized sequences (on-demand)

## 3. Data Pipeline

### Ingestion Job

```bash
python lakehouse/ingest_raw.py
```

Reads CSV and uploads to MinIO raw/

### Curation Job

```bash
python lakehouse/curate_text.py
```

- Tokenize, lowercase
- Remove special characters
- Save to curated/ as Parquet

### Kubernetes Job

```bash
kubectl apply -f k8s/jobs.yaml
```

## 4. Feature Store (Feast)

### Configuration

```yaml
# feature_repo/feature_store.yaml
project: spam_detection
registry: data/registry.db
provider: local
online_store:
  type: redis
  connection_string: redis:6379
```

### Features

Defined in `feature_repo/features.py`:
- Entity: `message_id`
- Features: `text_length`, `word_count`, `has_url`, `has_email`

### Materialize to Online Store

```python
from feast import FeatureStore
store = FeatureStore(repo_path="feature_repo")
store.materialize_incremental()
```

## 5. Distributed Training (Ray + MLflow)

### Training Script

```bash
python training/train_spam.py
```

Features:
- **Ray Distribute**: Parallel training across nodes
- **Ray Tune**: Hyperparameter optimization
  - `embedding_dim`: [16, 32, 64]
  - `lstm_units`: [20, 50, 100]
  - `dropout`: [0.1, 0.5]
  - `batch_size`: [16, 32, 64]
  - `epochs`: [10, 20, 30]
- **MLflow Logging**: Metrics, artifacts, model versioning

### Kubernetes Job

```bash
kubectl apply -f k8s/training-job.yaml
```

## 6. Model Registry & Serving

### MLflow Setup

```bash
kubectl apply -f k8s/mlflow.yaml
```

- **Backend**: SQLite (or Postgres for production)
- **Artifacts**: MinIO S3
- **UI**: http://mlflow:5000

### Promote Model

```python
import mlflow

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="spam_detection",
    version=1,
    stage="Production"
)
```

### Ray Serve API

```bash
python serving/serve_spam.py
```

Endpoint: `POST /api/predict`

```json
{
  "text": "Click here to win FREE money!"
}
```

Response:

```json
{
  "prediction": "spam",
  "confidence": 0.95
}
```

### Deploy to Kubernetes

```bash
# Build image
docker build -t spam-detection:latest .

# Deploy
kubectl apply -f k8s/ray-serve.yaml
```

Access via:
- Service: `ray-serve-spam:8000`
- LoadBalancer: `<external-ip>:8000`
- Ingress: `http://spam-api.local`

## 7. GitOps Deployment (Argo CD)

### Install Argo CD

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

### Create Application

```bash
kubectl apply -f argocd/spam-detection-app.yaml
```

### Progressive Delivery (Argo Rollouts)

```bash
kubectl apply -f argocd/ray-serve-rollout.yaml
```

Canary deployment:
1. 10% traffic → observe metrics
2. 50% traffic → if healthy
3. 100% traffic → full release

Automatic rollback if error rate > 5% or latency > 100ms

## 8. Monitoring & Observability

### Prometheus + Grafana

```bash
kubectl apply -f k8s/monitoring.yaml
```

Access:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Key Metrics

- API latency (p50, p95, p99)
- Error rate
- Throughput (requests/sec)
- Ray cluster health
- Pod CPU/memory usage

### Drift Detection

```bash
python lakehouse/monitor_drift.py
```

Runs every 6 hours (configurable):
- Compares production data stats to baseline
- Detects distribution shifts in text length, URL count, etc.
- Triggers retrain if drift exceeds threshold

## 9. Complete Workflow

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run notebook (existing)
jupyter notebook LSTM_detection_spam.ipynb

# 3. Extract training logic to train_spam.py
```

### Deployment to Kubernetes

```bash
# 1. Deploy infrastructure
kubectl apply -f k8s/minio.yaml
kubectl apply -f k8s/mlflow.yaml

# 2. Run data pipeline
kubectl apply -f k8s/jobs.yaml

# 3. Train model
kubectl apply -f k8s/training-job.yaml

# 4. Deploy API
docker build -t spam-detection:latest .
docker push <registry>/spam-detection:latest
kubectl apply -f k8s/ray-serve.yaml

# 5. Enable GitOps
kubectl apply -f argocd/spam-detection-app.yaml

# 6. Monitor
kubectl apply -f k8s/monitoring.yaml
kubectl apply -f k8s/drift-cronjob.yaml
```

### Production Checklist

- [ ] Kubernetes cluster configured
- [ ] Container registry (Harbor/Docker) ready
- [ ] Git repository initialized
- [ ] MinIO buckets created
- [ ] MLflow backend (Postgres) provisioned
- [ ] Redis for Feast online store
- [ ] Argo CD installed & connected
- [ ] Prometheus/Grafana dashboards created
- [ ] Alerting rules configured
- [ ] Backup strategy for MLflow registry
- [ ] RBAC & network policies configured

## 10. Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Lake | MinIO | S3-compatible object storage |
| Data Format | Delta Lake / Parquet | ACID transactions & versioning |
| Feature Store | Feast | Online/offline feature consistency |
| Training | Ray Train + Ray Tune | Distributed training & HPO |
| Tracking | MLflow | Model versioning & registry |
| Serving | Ray Serve | Scalable inference API |
| Orchestration | Argo CD | GitOps continuous deployment |
| Rollout | Argo Rollouts | Progressive delivery & canary |
| Monitoring | Prometheus/Grafana | Metrics & visualization |
| CI/CD | GitLab/GitHub Actions | Automated pipeline |

## 11. Next Steps

1. **Customize Hyperparameters**: Adjust ranges in `training/train_spam.py`
2. **Add Custom Metrics**: Extend MLflow logging with business KPIs
3. **Set Up Alerts**: Create Grafana alerts for SLA violations
4. **Configure Autoscaling**: Enable HPA for Ray workers based on queue depth
5. **Data Retention**: Set up MinIO lifecycle policies
6. **Cost Optimization**: Monitor resource utilization, adjust requests/limits

## References

- [Feast Documentation](https://docs.feast.dev/)
- [Ray Documentation](https://docs.ray.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Argo CD Documentation](https://argo-cd.readthedocs.io/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/)