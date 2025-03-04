from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import re
import scipy.special
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load FinBERT tokenizer and model
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def clean_text(text):
    """Clean text by removing numbers, special characters, and stop words."""
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if len(word) > 1 and word not in ENGLISH_STOP_WORDS and word not in ['eur', 'mn']])
    return text

def analyze_finbert(text):
    """Perform sentiment analysis using FinBERT."""
    if not text or text.strip() == "":
        return "neutral"  # Default to neutral for empty/missing text
    
    with torch.no_grad():
        input_sequence = finbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        logits = finbert_model(**input_sequence).logits
        scores = {
            k: v
            for k, v in zip(
                finbert_model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze()),
            )
        }
        return max(scores, key=scores.get).lower()

def analyze_dataset_with_finbert(csv_file):
    """Apply FinBERT sentiment analysis to a dataset with cleaned text."""
    df = pd.read_csv(csv_file, encoding='ISO-8859-1')

    # Detect if the dataset contains 'Text' or 'content' column
    text_column = None
    if "Text" in df.columns:
        text_column = "Text"
    elif "content" in df.columns:
        text_column = "content"
    else:
        raise ValueError("❌ Error: No 'Text' or 'content' column found in the CSV file.")

    # Clean text before applying sentiment analysis
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Apply FinBERT sentiment analysis
    df['FinBERT_Sentiment'] = df['cleaned_text'].apply(analyze_finbert)

    # Save output with input file name
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_file = f"{base_name}_finbert_output.csv"
    
    # Ensure important columns are kept in the final output
    save_columns = [col for col in ["title", "description", "content", "publishedAt", "url", "Text", "FinBERT_Sentiment"] if col in df.columns]
    df[save_columns].to_csv(output_file, index=False)

    print(f"✅ Results saved to: {output_file}")
    return df, output_file

def evaluate_finbert(df):
    """Compare FinBERT predictions with actual labels (if available)."""
    if "Sentiment" not in df.columns:
        print("⚠ No actual sentiment labels found. Skipping evaluation.")
        return
    
    sentiment_mapping = {"negative": -1, "neutral": 0, "positive": 1}
    df['Actual_Sentiment'] = df['Sentiment'].map(sentiment_mapping)
    df['Predicted_Sentiment'] = df['FinBERT_Sentiment'].map(sentiment_mapping)

    # Drop NaN values before evaluation
    df = df.dropna(subset=['Actual_Sentiment', 'Predicted_Sentiment'])

    # Compute accuracy & classification report
    accuracy = accuracy_score(df['Actual_Sentiment'], df['Predicted_Sentiment'])
    report = classification_report(df['Actual_Sentiment'], df['Predicted_Sentiment'], target_names=["Negative", "Neutral", "Positive"])
    
    print(f"\n✅ Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)

    # Confusion matrix visualization
    cm = confusion_matrix(df['Actual_Sentiment'], df['Predicted_Sentiment'])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Example usage:
csv_file = "scraped_news_7.csv"  # Change this to your dataset file
df, saved_file = analyze_dataset_with_finbert(csv_file)
evaluate_finbert(df)  # Only evaluates if 'Sentiment' exists
