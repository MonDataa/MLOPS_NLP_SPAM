from ray import serve
import ray
import mlflow.pytorch
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

serve.start()

@serve.deployment
class SpamPredictor:
    def __init__(self, model_uri, tokenizer_uri):
        # Load model from MLflow
        self.model = mlflow.keras.load_model(model_uri)
        
        # Load tokenizer
        with open(tokenizer_uri, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        self.max_len = 50
    
    async def __call__(self, request):
        text = request["text"]
        
        # Tokenize & pad
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post', truncating='post')
        
        # Predict
        pred = self.model.predict(padded)[0][0]
        label = "spam" if pred > 0.5 else "ham"
        
        return {
            "text": text,
            "prediction": label,
            "confidence": float(pred)
        }

# Deploy
predictor = SpamPredictor(
    model_uri="models:/spam_detection/production",
    tokenizer_uri="/path/to/tokenizer.pkl"
)

if __name__ == "__main__":
    ray.init()
    serve.run(predictor, port=8000)