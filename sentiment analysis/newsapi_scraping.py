import requests
import pandas as pd

# 🔹 Your API Key (Get from https://newsapi.org)
API_KEY = "cb3b9e55d4be42a0a0030173754f17ac"  # Replace with your actual API Key

# 🔹 Query with multiple startup sectors
QUERY = (
    "smart cities OR autonomous vehicles OR electric vehicles OR urban mobility OR "
    "public transport innovation OR smart infrastructure OR IoT in smart cities OR "
    "5G technology OR AI in urban planning"
)




# 🔹 API URL
URL = f"https://newsapi.org/v2/everything?q={QUERY}&language=en&sortBy=publishedAt&apiKey={API_KEY}"

def fetch_news():
    """Fetches news articles from NewsAPI and saves them to 'scraped_news.csv'."""
    response = requests.get(URL)
    
    # 🔹 Handle API Errors
    if response.status_code != 200:
        print(f"❌ API Error: {response.status_code} - {response.json().get('message', 'Unknown error')}")
        return None

    news_data = response.json()
    
    # 🔹 Check if articles exist
    if "articles" in news_data and news_data["articles"]:
        articles = news_data["articles"]
        df = pd.DataFrame(articles)[["title", "description", "content", "publishedAt", "url"]]
        
        # 🔹 Save CSV as 'scraped_news.csv' (overwrites previous file)
        filename = "scraped_news_7.csv"
        df.to_csv(filename, index=False)
        print(f"✅ News Data Saved to {filename}")
        return filename
    else:
        print("❌ No articles found.")
        return None

# 🔹 Run the news scraping function
if __name__ == "__main__":
    fetch_news()
