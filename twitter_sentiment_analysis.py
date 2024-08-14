import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re

# Step 1: Load Tweets from CSV
def load_tweets_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['tweet'].tolist()  # Assuming the column with tweets is named 'tweet'

# Step 2: Clean and Preprocess Tweets
def clean_tweet(tweet):
    # Remove mentions, hashtags, and links
    tweet = re.sub(r'@\w+|#\w+|http\S+', '', tweet)
    # Remove special characters and digits
    tweet = re.sub(r'\W|\d', ' ', tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    return tweet

def preprocess_tweets(tweets):
    return [clean_tweet(tweet) for tweet in tweets]

# Step 3: Perform Sentiment Analysis
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Step 4: Prepare Data for Model Training
def prepare_data(tweets, sentiments):
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(tweets).toarray()
    y = pd.get_dummies(sentiments)['positive']  # Convert to binary classification (positive vs. not positive)
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer

# Step 5: Train the Model
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Step 6: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Step 7: Save the Model and Vectorizer
def save_model(model, vectorizer):
    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(vectorizer, 'count_vectorizer.pkl')

# Step 8: Load and Use the Model for Prediction
def load_model():
    model = joblib.load('logistic_regression_model.pkl')
    vectorizer = joblib.load('count_vectorizer.pkl')
    return model, vectorizer

def predict_sentiment(model, vectorizer, tweet):
    tweet = clean_tweet(tweet)
    tweet_vectorized = vectorizer.transform([tweet]).toarray()
    return model.predict(tweet_vectorized)

# Main function to run the process
if __name__ == "__main__":
    file_path = "tweets.csv"  # Replace with your CSV file path
    tweets = load_tweets_from_csv(file_path)
    tweets_cleaned = preprocess_tweets(tweets)
    sentiments = [analyze_sentiment(tweet) for tweet in tweets_cleaned]

    # Prepare data for model training
    (X_train, X_test, y_train, y_test), vectorizer = prepare_data(tweets_cleaned, sentiments)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Save the model and vectorizer for future use
    save_model(model, vectorizer)

    # Example of loading and using the model for prediction
    model_loaded, vectorizer_loaded = load_model()
    sample_tweet = "I love using ChatGPT for my projects!"
    print(f"Predicted sentiment: {predict_sentiment(model_loaded, vectorizer_loaded, sample_tweet)}")
