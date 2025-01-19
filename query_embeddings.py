# import faiss
# import numpy as np
# import json
# from sentence_transformers import SentenceTransformer

# # Load FAISS index and metadata
# index = faiss.read_index("tweets.faiss")
# with open("tweets_metadata.json", "r") as f:
#     tweets = json.load(f)

# # Load embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # User query
# query = input("Enter your query: ")
# query_vector = model.encode(query).reshape(1, -1).astype("float32")

# # Search for similar embeddings
# distances, indices = index.search(query_vector, k=5)
# retrieved_tweets = [tweets[i] for i in indices[0]]

# print("\nTop Relevant Tweets:")
# for tweet in retrieved_tweets:
#     print("-", tweet)

import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load FAISS index and metadata
index = faiss.read_index("tweets.faiss")
with open("tweets_metadata.json", "r") as f:
    tweets = json.load(f)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_relevant_tweets(query, k=5):
    """
    Retrieve the top k relevant tweets for a given query.

    Args:
        query (str): The user query.
        k (int): Number of top relevant tweets to retrieve.

    Returns:
        list: List of top relevant tweets.
    """
    query_vector = model.encode(query).reshape(1, -1).astype("float32")
    distances, indices = index.search(query_vector, k)
    return [tweets[i] for i in indices[0]]
