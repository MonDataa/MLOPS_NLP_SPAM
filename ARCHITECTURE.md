create_directory: data/, lakehouse/, feature_repo/, training/, serving/, mlflow/, k8s/, argocd/, .azure/

create_files:
  lakehouse:
    - ingest_raw.py: Load CSV → MinIO raw/
    - curate_text.py: Clean text → MinIO curated/ (Parquet)
    - monitor_drift.py: Detect feature distribution shifts

  feature_repo:
    - features.py: Feast entity & feature definitions
    - feature_store.yaml: Feast configuration (Redis online store)

  training:
    - train_spam.py: Ray Train + Ray Tune + MLflow logging
      - Tokenizer, padding, train/val split
      - LSTM model definition
      - Hyperparameter optimization
      - Model & tokenizer artifact logging

  serving:
    - serve_spam.py: Ray Serve API
      - Load model from MLflow registry
      - Load tokenizer artifact
      - POST /api/predict endpoint
      - Preprocessing & prediction logic

  k8s:
    - minio.yaml: Object storage (S3-compatible)
    - mlflow.yaml: Model tracking & registry
    - jobs.yaml: Ingestion & curation batch jobs
    - training-job.yaml: Training job (Ray)
    - ray-serve.yaml: Inference deployment (2 replicas)
    - monitoring.yaml: Prometheus + Grafana
    - drift-cronjob.yaml: Drift detection every 6 hours

  argocd:
    - spam-detection-app.yaml: Argo CD application manifest
    - ray-serve-rollout.yaml: Canary deployment (10% → 50% → 100%)

  root:
    - requirements.txt: All Python dependencies
    - Dockerfile: Container image for serving
    - README.md: Complete architecture & deployment guide
    - IMPLEMENTATION_GUIDE.md: Step-by-step roadmap (8-12 weeks)

architecture_layers:
  1_data_layer:
    storage: MinIO (S3)
    format: Parquet + Delta Lake
    buckets: raw/, curated/, features/
    job_orchestration: Kubernetes Jobs

  2_feature_layer:
    online_store: Redis
    offline_store: Parquet/Delta
    framework: Feast
    entity: message_id
    features: text_length, word_count, has_url, has_email

  3_training_layer:
    framework: Ray Train + Ray Tune (distributed HPO)
    tracking: MLflow (metrics, models, artifacts)
    model: LSTM/BiLSTM for binary classification
    optimization_params:
      - embedding_dim: [16, 32, 64]
      - lstm_units: [20, 50, 100]
      - dropout: [0.1, 0.5]
      - batch_size: [16, 32, 64]
      - epochs: [10, 20, 30]

  4_serving_layer:
    api_framework: Ray Serve
    deployment: Kubernetes (replicas: 2)
    endpoint: POST /api/predict
    input: {"text": "..."}
    output: {"prediction": "spam/ham", "confidence": 0.95}

  5_orchestration_layer:
    gitops: Argo CD (GitOps continuous deployment)
    progressive_delivery: Argo Rollouts (canary: 10% → 50% → 100%)
    strategy: Auto-sync on git push
    rollback_trigger: Error rate > 5%, latency > 100ms

  6_monitoring_layer:
    metrics: Prometheus
    visualization: Grafana
    drift_detection: Monitor feature stats vs baseline
    alerting: Custom thresholds for SLAs
    logs: ELK/Loki (optional)

data_flow:
  ingestion: CSV → MinIO raw/
  curation: raw/ → text_cleaning → curated/ (Parquet)
  features: curated/ → feature_engineering → features/ (Delta)
  feast: curated/ → Feast offline → Redis online
  training: features/ + Feast → Ray Train → MLflow Registry
  serving: MLflow Registry → Ray Serve → API
  monitoring: API → Prometheus → Grafana
  drift: API logs → Monitor Drift CronJob → Trigger Retrain

deployment_order:
  1. Infrastructure: MinIO, MLflow, Redis
  2. Ingestion: ingest_raw.py job
  3. Curation: curate_text.py job
  4. Training: train_spam.py job
  5. Serving: ray-serve deployment
  6. GitOps: Argo CD application
  7. Monitoring: Prometheus/Grafana + Drift CronJob

scalability:
  ray_workers: HPA based on queue depth
  serving_replicas: 2-10 (HPA based on request rate)
  data_pipeline: Batched jobs (daily/hourly)
  feature_materialization: Every 6 hours

reliability:
  storage_backup: MinIO versioning + regular snapshots
  model_versioning: MLflow registry (all runs preserved)
  rolling_deployment: Argo Rollouts canary strategy
  monitoring_alerts: SLA breaches trigger ops

notebook_migration:
  existing_notebook: LSTM_detection_spam.ipynb
  convert_to:
    data_loading: lakehouse/ingest_raw.py
    preprocessing: lakehouse/curate_text.py (text cleaning)
    preprocessing: training/train_spam.py (tokenizer, padding)
    model_definition: training/train_spam.py
    training_loop: training/train_spam.py (with MLflow)
    inference: serving/serve_spam.py

next_phases_beyond_initial:
  phase_9_batch_inference: Scheduled batch predictions
  phase_10_multi_model: A/B testing framework
  phase_11_explainability: SHAP/LIME integration
  phase_12_retraining_automation: Trigger on drift + performance degradation