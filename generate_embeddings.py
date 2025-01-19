from sentence_transformers import SentenceTransformer
import pandas as pd
import json

# Load the pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load preprocessed dataset
df = pd.read_csv("elon_musk_cleaned_tweets.csv")

# Ensure all rows in the 'clean_text' column are strings
df["clean_text"] = df["clean_text"].fillna("").astype(str)

# Generate embeddings
embeddings = []
for tweet in df["clean_text"]:
    if tweet.strip():  # Skip empty tweets
        embedding = model.encode(tweet).tolist()
        embeddings.append({
            "tweet": tweet,
            "embedding": embedding
        })

# Save embeddings to a JSON file
with open("tweet_embeddings.json", "w") as f:
    json.dump(embeddings, f)

print("Embeddings generated and saved to tweet_embeddings.json")
