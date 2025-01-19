import pandas as pd
import re

# Load dataset
df = pd.read_csv("elon_musk_tweets.csv")  # Replace with your dataset file name

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = text.strip()  # Remove extra spaces
    return text

# Apply preprocessing
df["clean_text"] = df["text"].apply(preprocess_text)

# Save cleaned data
df.to_csv("elon_musk_cleaned_tweets.csv", index=False)
print("Preprocessed tweets saved to elon_musk_cleaned_tweets.csv")
