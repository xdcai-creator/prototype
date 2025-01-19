import faiss
import numpy as np
import json

# Load embeddings
with open("tweet_embeddings.json", "r") as f:
    embeddings = json.load(f)

# Extract vectors and metadata
vectors = np.array([item["embedding"] for item in embeddings]).astype("float32")
tweets = [item["tweet"] for item in embeddings]

# Create FAISS index
index = faiss.IndexFlatL2(vectors.shape[1])  # L2 distance for similarity
index.add(vectors)

# Save index and metadata
faiss.write_index(index, "tweets.faiss")
with open("tweets_metadata.json", "w") as f:
    json.dump(tweets, f)

print("Embeddings stored in FAISS index and metadata saved.")
