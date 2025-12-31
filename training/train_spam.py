import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.tensorflow import TensorflowTrainer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import boto3
from botocore.client import Config
import mlflow
import mlflow.keras
import os

# MinIO config
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

# MLflow config
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("spam_detection")

def load_data():
    s3.download_file(BUCKET_NAME, 'curated/sms_clean.parquet', '/tmp/sms_clean.parquet')
    df = pd.read_parquet('/tmp/sms_clean.parquet')
    # Assume label is 0/1
    X = df['message']
    y = df['label'].map({'ham': 0, 'spam': 1})
    return X, y

def preprocess_data(X, y):
    vocab_size = 500
    max_len = 50
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded, y.values, tokenizer

def build_model(vocab_size=500, embedding_dim=16, lstm_units=20, dropout=0.2, max_len=50):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(LSTM(lstm_units, dropout=dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_func(config):
    # Load and preprocess data
    X, y = load_data()
    X_padded, y_labels, tokenizer = preprocess_data(X, y)
    
    # Split train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_padded, y_labels, test_size=0.2, random_state=42)
    
    # Build model
    model = build_model(
        embedding_dim=config["embedding_dim"],
        lstm_units=config["lstm_units"],
        dropout=config["dropout"],
        max_len=50
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=0
    )
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_params(config)
        mlflow.log_metric("val_accuracy", max(history.history['val_accuracy']))
        mlflow.keras.log_model(model, "model")
        # Save tokenizer
        import pickle
        with open('/tmp/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        mlflow.log_artifact('/tmp/tokenizer.pkl')

# Ray Tune config
from ray import tune
from ray.tune.schedulers import ASHAScheduler

config = {
    "embedding_dim": tune.choice([16, 32, 64]),
    "lstm_units": tune.choice([20, 50, 100]),
    "dropout": tune.uniform(0.1, 0.5),
    "epochs": tune.choice([10, 20, 30]),
    "batch_size": tune.choice([16, 32, 64])
}

scheduler = ASHAScheduler(
    max_t=30,
    grace_period=5,
    reduction_factor=2
)

tuner = tune.Tuner(
    train_func,
    tune_config=tune.TuneConfig(
        metric="val_accuracy",
        mode="max",
        scheduler=scheduler,
        num_samples=10
    ),
    param_space=config
)

results = tuner.fit()
best_result = results.get_best_result()
print("Best config:", best_result.config)