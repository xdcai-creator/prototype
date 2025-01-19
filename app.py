# from flask import Flask, request, jsonify, render_template
# import faiss
# import numpy as np
# import json
# from sentence_transformers import SentenceTransformer
# from transformers import AutoModelForCausalLM, AutoTokenizer

# app = Flask(__name__)

# # Load FAISS index and metadata
# index = faiss.read_index("tweets.faiss")
# with open("tweets_metadata.json", "r") as f:
#     tweets = json.load(f)

# # Load embedding model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Load GPT-Neo model and tokenizer
# model_name = "EleutherAI/gpt-neo-1.3B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Assign the eos_token as the pad_token to resolve padding issues
# tokenizer.pad_token = tokenizer.eos_token


# def get_relevant_tweets(query):
#     """
#     Retrieve the most relevant tweets for the given query using FAISS.
#     Args:
#         query (str): User query.

#     Returns:
#         list: Top 5 relevant tweets.
#     """
#     query_vector = embedding_model.encode(query).reshape(1, -1).astype("float32")
#     distances, indices = index.search(query_vector, k=5)
#     retrieved_tweets = [tweets[i] for i in indices[0]]
#     return retrieved_tweets


# def generate_response(context, query):
#     """
#     Generate a response using GPT-Neo based on the provided context and query.
#     Args:
#         context (str): Context text summarizing relevant information.
#         query (str): User query or question to be answered.

#     Returns:
#         str: Generated response from the model.
#     """
#     input_text = (
#         f"The following tweets summarize Elon Musk's thoughts:\n"
#         f"{context[:300]}  \n\n"
#         f"Based on these tweets, provide a clear, accurate, and concise response.\n"
#         f"Your response must summarize the tweets in relation to the question and be no longer than 1-2 sentences.\n"
#         f"Question: {query}\n"
#         f"Answer:"
#     )
#     inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
#     outputs = model.generate(
#         inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         max_new_tokens=40,
#         temperature=0.7,
#         top_p=0.9,
#         repetition_penalty=2.0,
#         pad_token_id=tokenizer.pad_token_id,
#         do_sample=True,
#     )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response.strip()


# @app.route('/')
# def home():
#     """Render the homepage."""
#     return render_template('index.html')


# @app.route('/query', methods=['POST'])
# def query():
#     """
#     Handle user query.
#     - Retrieve relevant tweets using FAISS.
#     - Generate a response using GPT-Neo.
#     """
#     data = request.get_json()
#     if not data or 'question' not in data:
#         return jsonify({"error": "Missing 'question' in request body"}), 400

#     question = data['question']
#     relevant_tweets = get_relevant_tweets(question)
#     context = "\n".join(relevant_tweets)
#     response = generate_response(context, question)

#     return jsonify({
#         "relevant_tweets": relevant_tweets,
#         "response": response
#     })


# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
from query_embeddings import get_relevant_tweets
from generate_response import generate_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided."}), 400

    # Get relevant tweets
    relevant_tweets = get_relevant_tweets(question)

    # Concatenate tweets for context
    context = "\n".join([f"- {tweet}" for tweet in relevant_tweets])

    # Generate response
    response = generate_response(context, question)

    return jsonify({
        "question": question,
        "relevant_tweets": relevant_tweets,
        "answer": response
    })

if __name__ == "__main__":
    app.run(debug=True)
