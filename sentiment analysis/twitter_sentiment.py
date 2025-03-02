
import tweepy
import pandas as pd
import re
import os
from textblob import TextBlob

# Set up Twitter API credentials (replace with your keys)
API_KEY = "77HZZaDIuVShAuzNCVzsHQSMq"
API_SECRET = "jEK4Pr8Q8Z1EFjskgNRrFsXbCI9bnLTDdz6lai7VyHTUFNpgMd"
ACCESS_TOKEN = "1396837900721868800-BInb1CfJ5Bvs2gJI2YdRH6Tb8oigSX"
ACCESS_SECRET = "mcxsx2QE6vdJ1GNiOkGbWUygGG0822nEtvBKioBwuHC7G"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAADh1zgEAAAAAGPjxFNokkKG1miI4%2FcfeOKQIPss%3Dfi6mfcLl2VF40w4AXsEPtokSZlBMYpywzGtGUIF2dIEImQAdnq"  # Required for API v2

# Authenticate with Twitter API v2
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Function to clean tweet text
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)  # Remove mentions
    tweet = re.sub(r'#[A-Za-z0-9]+', '', tweet)  # Remove hashtags
    tweet = re.sub(r'RT : ', '', tweet)  # Remove retweet symbols
    return tweet.strip()

# Function to get sentiment score
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity  # -1 to 1 (negative to positive)

# Fetch tweets using Twitter API v2
def fetch_tweets(keyword, count=100):
    tweets_data = []
    
    query = f"{keyword} lang:en -is:retweet"  # Exclude retweets
    response = client.search_recent_tweets(query=query, max_results=min(count, 100), tweet_fields=["created_at", "text"])

    if response.data:
        for tweet in response.data:
            cleaned_tweet = clean_tweet(tweet.text)
            sentiment_score = get_sentiment(cleaned_tweet)
            tweets_data.append([tweet.created_at, cleaned_tweet, sentiment_score])

    df = pd.DataFrame(tweets_data, columns=['Timestamp', 'Tweet', 'Sentiment'])
    return df

# Example usage
if __name__ == "__main__":
    keyword = "startup investment"
    df = fetch_tweets(keyword, count=100)
    df.to_csv("startup_sentiment.csv", index=False)
    print("Scraped and saved tweets successfully!")

