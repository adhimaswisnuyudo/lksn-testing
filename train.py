import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import boto3
import os
print("Files in current dir:", os.listdir())


# Load dataset
df = pd.read_csv('product_reviews.csv')
X = df['review']
y = df['sentiment']

# Vectorize
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/sentiment_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

# Upload to S3
s3 = boto3.client('s3')
bucket = os.environ['S3_BUCKET']
s3.upload_file('model/sentiment_model.pkl', bucket, 'sentiment_model.pkl')
s3.upload_file('model/vectorizer.pkl', bucket, 'vectorizer.pkl')
