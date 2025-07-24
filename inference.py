import joblib
import boto3
import os

# Ambil file model dan vectorizer dari local atau S3
def download_model_from_s3(bucket, model_key, vectorizer_key, local_dir='model'):
    s3 = boto3.client('s3')
    os.makedirs(local_dir, exist_ok=True)

    model_path = os.path.join(local_dir, 'sentiment_model.pkl')
    vectorizer_path = os.path.join(local_dir, 'vectorizer.pkl')

    s3.download_file(bucket, model_key, model_path)
    s3.download_file(bucket, vectorizer_key, vectorizer_path)

    return model_path, vectorizer_path

# Prediksi sentimen
def predict_sentiment(texts, model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    return predictions

if __name__ == "__main__":
    # Contoh penggunaan
    BUCKET = os.environ.get('S3_BUCKET', 'lkse2025-sentiment-jabar')
    MODEL_KEY = 'sentiment_model.pkl'
    VECTORIZER_KEY = 'vectorizer.pkl'

    # Download model dan vectorizer dari S3
    model_path, vectorizer_path = download_model_from_s3(BUCKET, MODEL_KEY, VECTORIZER_KEY)

    # Input teks dari user (bisa diganti dengan API / input eksternal)
    sample_texts = [
        "Produk ini sangat berkualitas dan pengirimannya cepat",
        "Saya kecewa, barangnya rusak dan tidak sesuai gambar"
    ]

    results = predict_sentiment(sample_texts, model_path, vectorizer_path)

    for text, sentiment in zip(sample_texts, results):
        print(f"Ulasan: {text}\nSentimen: {sentiment}\n")
