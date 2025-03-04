import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URL for TechCrunch Startups section
BASE_URL = "https://techcrunch.com/startups/"

# Keywords for filtering
KEYWORDS = ["AI", "Tech", "Startup", "Investment", "Funding", "Business", "Innovation"]


def get_techcrunch_articles():
    """Scrape latest startup articles from TechCrunch."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(BASE_URL, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve TechCrunch page. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("div", class_="post-block")  # Updated selector

    scraped_data = []
    for article in articles:
        title_element = article.find("a", class_="post-block__title__link")
        if not title_element:
            continue  # Skip if title isn't found

        title = title_element.text.strip()
        link = title_element["href"]

        content = get_article_content(link)

        if any(keyword.lower() in title.lower() or keyword.lower() in content.lower() for keyword in KEYWORDS):
            scraped_data.append({"Title": title, "URL": link, "Content": content})

    return scraped_data


def get_article_content(url):
    """Extract full article content from TechCrunch."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return "Failed to fetch content."

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")  # Extract all paragraphs
    full_content = " ".join([p.text for p in paragraphs])
    return full_content

# Run scraper
articles = get_techcrunch_articles()
print(articles)  # Check if data is being extracted

if articles:  # Only save if data exists
    df = pd.DataFrame(articles)
    df.to_csv("techcrunch_startup_news.csv", index=False)
    print("✅ Scraping complete! Data saved.")
else:
    print("⚠ No articles found. Check the website structure or keyword filtering.")



