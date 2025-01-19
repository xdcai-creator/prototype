# from pyexpat import model
# import faiss
# import numpy as np
# import json
# from sentence_transformers import SentenceTransformer
# from tokenizers import Tokenizer
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load models
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# gpt_model_name = "gpt2"
# gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
# gpt_model = AutoModelForCausalLM.from_pretrained(gpt_model_name)

# # Load FAISS index and metadata
# index = faiss.read_index("tweets.faiss")
# with open("tweets_metadata.json", "r") as f:
#     tweets = json.load(f)

# def query_embeddings(query, top_k=5):
#     # Generate query embedding
#     query_vector = embedding_model.encode(query).reshape(1, -1).astype("float32")

#     # Search for similar embeddings
#     distances, indices = index.search(query_vector, k=top_k)
#     retrieved_tweets = [tweets[i] for i in indices[0]]

#     return retrieved_tweets

# def generate_response(context, query):
#     # Combine the context and query
#     input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
#     # Generate response with parameters to control repetition
#     inputs = Tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
#     outputs = model.generate(
#         inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],  # Explicitly set the attention mask
#         max_length=150,  # Limit response length
#         temperature=0.7,  # Add randomness to responses
#         top_p=0.9,  # Use nucleus sampling
#         repetition_penalty=1.2,  # Penalize repeated phrases
#         pad_token_id=Tokenizer.eos_token_id  # Set padding token ID
#     )

#     # Decode and return the response
#     response = Tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# # Main script
# if __name__ == "__main__":
#     query = input("Enter your question: ")

#     # Retrieve relevant tweets
#     relevant_tweets = query_embeddings(query)
#     context = "\n".join([f"- {tweet}" for tweet in relevant_tweets])


#     # Generate response using GPT
#     response = generate_response(context, query)

#     print("\nTop Relevant Tweets:")
#     for tweet in relevant_tweets:
#         print("-", tweet)

#     print("\nGenerated Response:")
#     print(response)

# import faiss
# import numpy as np
# import json
# from sentence_transformers import SentenceTransformer
# from generate_response import generate_response

# # Load models
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Load FAISS index and metadata
# index = faiss.read_index("tweets.faiss")
# with open("tweets_metadata.json", "r") as f:
#     tweets = json.load(f)

# def query_embeddings(query, top_k=5):
#     # Generate query embedding
#     query_vector = embedding_model.encode(query).reshape(1, -1).astype("float32")

#     # Search for similar embeddings
#     distances, indices = index.search(query_vector, k=top_k)
#     retrieved_tweets = [tweets[i] for i in indices[0]]
#     return retrieved_tweets

# if __name__ == "__main__":
#     query = input("Enter your question: ")

#     # Retrieve relevant tweets
#     relevant_tweets = query_embeddings(query)
#     context = "\n".join([f"- {tweet}" for tweet in relevant_tweets])

#     # Generate GPT-based response
#     response = generate_response(context, query)

#     # Display results
#     print("\nTop Relevant Tweets:")
#     for tweet in relevant_tweets:
#         print("-", tweet)

#     print("\nGenerated Response:")
#     print(response)

import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from generate_response import generate_response

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and metadata
index = faiss.read_index("tweets.faiss")
with open("tweets_metadata.json", "r") as f:
    tweets = json.load(f)

def query_embeddings(query, top_k=5):
    # Generate query embedding
    query_vector = embedding_model.encode(query).reshape(1, -1).astype("float32")

    # Search for similar embeddings
    distances, indices = index.search(query_vector, k=top_k)
    retrieved_tweets = [tweets[i] for i in indices[0]]
    return retrieved_tweets

if __name__ == "__main__":
    # Input from the user
    query = input("Enter your question: ")

    # Retrieve relevant tweets
    relevant_tweets = query_embeddings(query)

    # Limit context to top 3 tweets for clarity
    context = " ".join(relevant_tweets[:3])

    # Generate GPT-based response
    response = generate_response(context, query)

    # Display results
    print("\nTop Relevant Tweets:")
    for tweet in relevant_tweets[:3]:
        print("-", tweet)

    print("\nGenerated Response:")
    print(response)
