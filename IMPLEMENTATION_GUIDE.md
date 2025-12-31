# MLOPS NLP Spam Detection - Implementation Guide

Roadmap for transforming your LSTM spam detection notebook into a production-grade MLOps system.

## Phase 1: Foundation (Weeks 1-2)

### 1.1 Local Development Setup
- [ ] Set up local Kubernetes (k3s recommended)
- [ ] Install kubectl, helm
- [ ] Configure local Docker registry or use Harbor
- [ ] Initialize Git repository

### 1.2 Deploy Infrastructure
- [ ] Deploy MinIO (object storage)
- [ ] Deploy MLflow (model tracking)
- [ ] Deploy Redis (for Feast online store)

**Commands:**
```bash
kubectl apply -f k8s/minio.yaml
kubectl apply -f k8s/mlflow.yaml
kubectl apply -f k8s/redis.yaml  # Create if not exists
```

### 1.3 Extract Notebook Logic
- [ ] Convert notebook cells to modular Python scripts
- [ ] Create `lakehouse/ingest_raw.py`
- [ ] Create `lakehouse/curate_text.py`
- [ ] Define preprocessing as reusable functions

**Mapping:**
- Notebook data loading → `ingest_raw.py`
- Text cleaning (lower, regex) → `curate_text.py`
- Tokenizer & padding → `train_spam.py` (preprocessing module)

---

## Phase 2: Data Pipeline (Weeks 2-3)

### 2.1 Implement Ingestion
- [ ] Load CSV from local/cloud storage
- [ ] Upload to MinIO `s3://mlops/raw/sms_spam.csv`

**Code:** `lakehouse/ingest_raw.py`

### 2.2 Implement Curation
- [ ] Read from MinIO raw/
- [ ] Apply text cleaning (lowercase, remove special chars, etc.)
- [ ] Save to MinIO `curated/sms_clean.parquet`

**Code:** `lakehouse/curate_text.py`

### 2.3 Create Kubernetes Jobs
- [ ] Define ingest job manifest
- [ ] Define curate job manifest
- [ ] Test job execution

**File:** `k8s/jobs.yaml`

### 2.4 Feature Engineering
- [ ] Extract features: `text_length`, `word_count`, `has_url`, `has_email`
- [ ] Store in `features/nlp_features.parquet`

---

## Phase 3: Feature Store (Weeks 3-4)

### 3.1 Set Up Feast
- [ ] Initialize feature store repo
- [ ] Define entity: `message_id`
- [ ] Define feature view: `message_features`

**File:** `feature_repo/feature_store.yaml`

### 3.2 Offline Features
- [ ] Load curated data
- [ ] Compute features
- [ ] Store in Delta Lake or Parquet

### 3.3 Online Features
- [ ] Deploy Redis as online store
- [ ] Materialize features to Redis
- [ ] Test get_online_features API

---

## Phase 4: Training Pipeline (Weeks 4-6)

### 4.1 Refactor Notebook Model
- [ ] Extract model definition (Sequential LSTM)
- [ ] Extract training logic
- [ ] Add MLflow instrumentation

**Key Components:**
```python
def build_model(vocab_size, embedding_dim, lstm_units, dropout, max_len):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(LSTM(lstm_units, dropout=dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

### 4.2 Add MLflow Logging
- [ ] Log hyperparameters
- [ ] Log metrics (accuracy, loss, val_accuracy)
- [ ] Log model artifact
- [ ] Log tokenizer.pkl

**Code:** `training/train_spam.py`

### 4.3 Implement Ray Train
- [ ] Distribute training across Ray workers
- [ ] Use Ray's TensorflowTrainer

### 4.4 Implement Ray Tune (HPO)
- [ ] Define search space:
  - `embedding_dim`: [16, 32, 64]
  - `lstm_units`: [20, 50, 100]
  - `dropout`: [0.1, 0.5]
  - `batch_size`: [16, 32, 64]
- [ ] Use ASHA scheduler for efficient pruning
- [ ] Track best run

### 4.5 Create Training Job
- [ ] Create Kubernetes job manifest
- [ ] Mount training script as ConfigMap
- [ ] Test local execution first

**File:** `k8s/training-job.yaml`

---

## Phase 5: Model Serving (Weeks 6-7)

### 5.1 Build Inference API
- [ ] Create Ray Serve deployment
- [ ] Load model from MLflow registry
- [ ] Load tokenizer
- [ ] Implement `/predict` endpoint

**Code:** `serving/serve_spam.py`

### 5.2 Containerize
- [ ] Create Dockerfile with all dependencies
- [ ] Test image locally

**File:** `Dockerfile`

### 5.3 Deploy to Kubernetes
- [ ] Create Ray Serve deployment manifest
- [ ] Expose via Service/Ingress
- [ ] Test endpoint

**File:** `k8s/ray-serve.yaml`

### 5.4 API Testing
```bash
curl -X POST http://spam-api.local/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Click here for FREE money!"}'
```

---

## Phase 6: Monitoring & Observability (Weeks 7-8)

### 6.1 Deploy Prometheus
- [ ] Configure scrape targets:
  - Ray Serve metrics (8265)
  - MLflow metrics (5000)
  - Kubernetes metrics

**File:** `k8s/monitoring.yaml`

### 6.2 Deploy Grafana
- [ ] Add Prometheus data source
- [ ] Create dashboards:
  - API latency (p50, p95, p99)
  - Error rate
  - Throughput
  - Model accuracy trend

### 6.3 Implement Drift Detection
- [ ] Compare production stats to baseline
- [ ] Alert on distribution shift
- [ ] Trigger retrain if needed

**Code:** `lakehouse/monitor_drift.py`

### 6.4 Create CronJob
- [ ] Run drift check every 6 hours
- [ ] Log results to monitoring system

**File:** `k8s/drift-cronjob.yaml`

---

## Phase 7: GitOps & CI/CD (Weeks 8-9)

### 7.1 Initialize Git Repository
- [ ] Create GitLab/GitHub repo
- [ ] Push code structure
- [ ] Set up branch protection rules

### 7.2 Install Argo CD
- [ ] Deploy Argo CD to cluster
- [ ] Configure access
- [ ] Create application manifest

**File:** `argocd/spam-detection-app.yaml`

### 7.3 Set Up CI/CD Pipeline
- [ ] Create `.gitlab-ci.yml` or GitHub Actions workflow
- [ ] Build image on commit
- [ ] Push to registry
- [ ] Auto-sync with Argo CD

**Pipeline Steps:**
1. Lint & test
2. Build Docker image
3. Push to registry
4. Argo CD detects new image tag
5. Auto-deploy to cluster

### 7.4 Implement Progressive Delivery
- [ ] Deploy Argo Rollouts
- [ ] Configure canary strategy: 10% → 50% → 100%
- [ ] Define rollback metrics

**File:** `argocd/ray-serve-rollout.yaml`

---

## Phase 8: Production Hardening (Weeks 9-10)

### 8.1 Infrastructure
- [ ] Configure persistent storage (PVC/Longhorn)
- [ ] Set up backup strategy for MLflow
- [ ] Configure network policies
- [ ] Implement RBAC

### 8.2 Security
- [ ] Enable TLS/SSL for all services
- [ ] Configure secret management (Vault/K8s Secrets)
- [ ] Implement API authentication/authorization
- [ ] Audit logging

### 8.3 Scalability
- [ ] Configure HPA for Ray workers
- [ ] Configure HPA for serving pods
- [ ] Load testing
- [ ] Optimize resource requests/limits

### 8.4 Reliability
- [ ] Configure liveness/readiness probes
- [ ] Implement circuit breaker pattern
- [ ] Configure pod disruption budgets
- [ ] Disaster recovery plan

---

## Implementation Checklist

### Data Layer
- [ ] MinIO deployed & buckets created (raw/, curated/, features/)
- [ ] Ingestion job completed & tested
- [ ] Curation job completed & tested
- [ ] Delta Lake or Parquet format validated

### Feature Store
- [ ] Feast installed & configured
- [ ] Entity & features defined
- [ ] Online store (Redis) operational
- [ ] Feature materialization working

### Training
- [ ] Model training script extracted from notebook
- [ ] MLflow integration complete
- [ ] Ray Train configured
- [ ] Ray Tune hyperparameter search operational
- [ ] Training job manifest created & tested

### Serving
- [ ] Ray Serve API implemented
- [ ] Tokenizer & model loading working
- [ ] Dockerfile created & tested
- [ ] Kubernetes deployment manifest created
- [ ] API endpoint accessible

### Monitoring
- [ ] Prometheus scraping metrics
- [ ] Grafana dashboards created
- [ ] Drift detection job running
- [ ] Alerts configured

### GitOps
- [ ] Git repository initialized
- [ ] Argo CD installed & configured
- [ ] Application manifest created
- [ ] CI/CD pipeline operational
- [ ] Argo Rollouts (canary) configured

### Production
- [ ] All security measures implemented
- [ ] High availability configured
- [ ] Backup & restore tested
- [ ] Load testing completed
- [ ] Documentation complete

---

## Key Files to Create/Modify

```
✅ Created:
├── k8s/minio.yaml
├── k8s/mlflow.yaml
├── k8s/jobs.yaml (ingestion/curation)
├── k8s/training-job.yaml
├── k8s/ray-serve.yaml
├── k8s/monitoring.yaml
├── k8s/drift-cronjob.yaml
├── lakehouse/ingest_raw.py
├── lakehouse/curate_text.py
├── lakehouse/monitor_drift.py
├── feature_repo/features.py
├── feature_repo/feature_store.yaml
├── training/train_spam.py
├── serving/serve_spam.py
├── argocd/spam-detection-app.yaml
├── argocd/ray-serve-rollout.yaml
├── requirements.txt
├── Dockerfile
└── README.md

📝 To Create:
├── .gitlab-ci.yml or .github/workflows/deploy.yml
├── k8s/redis.yaml (Feast online store)
├── k8s/postgres.yaml (MLflow backend for prod)
├── helm/values.yaml (Helm chart for easy deployment)
└── scripts/deploy.sh (Helper deployment script)
```

---

## Testing Strategy

### Local Testing
```bash
# 1. Test MinIO & data pipeline
python lakehouse/ingest_raw.py
python lakehouse/curate_text.py

# 2. Test training
python training/train_spam.py --epochs=1 --samples=100

# 3. Test serving
python serving/serve_spam.py
curl -X POST http://localhost:8000/api/predict -d '{"text": "test"}'
```

### Kubernetes Testing
```bash
# 1. Deploy all manifests
kubectl apply -f k8s/

# 2. Verify pods
kubectl get pods

# 3. Test endpoint
kubectl port-forward svc/ray-serve-spam 8000:8000
curl http://localhost:8000/api/predict
```

---

## Estimated Timeline

- **Phase 1**: 2 weeks (infrastructure)
- **Phase 2**: 1-2 weeks (data pipeline)
- **Phase 3**: 1-2 weeks (feature store)
- **Phase 4**: 2 weeks (training)
- **Phase 5**: 1 week (serving)
- **Phase 6**: 1-2 weeks (monitoring)
- **Phase 7**: 1 week (GitOps)
- **Phase 8**: 1-2 weeks (hardening)

**Total: ~10-12 weeks for production-ready system**

---

## Success Metrics

✅ System is ready when:
1. Data pipeline runs automatically on schedule
2. Model training completes and logs to MLflow
3. API serves predictions with <100ms latency at p95
4. Drift detection triggers retrain automatically
5. Canary deployment succeeds with zero errors
6. Monitoring captures all key metrics
7. System handles 1000 req/sec with <5% error rate
8. Complete documentation covers deployment & operations

---

For detailed implementation of each component, refer to code files in respective directories.