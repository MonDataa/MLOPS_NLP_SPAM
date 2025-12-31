FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/serve_spam.py .
COPY training/train_spam.py .

EXPOSE 8000 8265

CMD ["python", "serve_spam.py"]