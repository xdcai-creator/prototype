# Elon’s Tweets Chat Application

This project creates a web application that allows users to interact with Elon Musk's tweets as if they were chatting with him. The application leverages preprocessed tweet data, embeddings, and a language model to generate human-like responses to user queries.

---

## **Features**
- **Interactive Chat Interface**: Users can ask questions, and the application generates responses based on Elon Musk's tweets.
- **Tweet Retrieval**: Uses embeddings and FAISS for similarity search to retrieve relevant tweets.
- **Language Model Integration**: Generates conversational responses using the GPT-Neo model.
- **Modular Design**: Code is organized into separate modules for preprocessing, embedding generation, querying, and response generation.

---

## **Directory Structure**
```plaintext
prototype/
├── app.py                   # Main Flask application
├── generate_embeddings.py   # Script to generate embeddings from tweets
├── generate_response.py     # Module for generating responses
├── main.py                  # Script for local testing of response generation
├── preprocess_tweets.py     # Preprocessing module for cleaning tweets
├── query_embeddings.py      # Handles FAISS-based query matching
├── store_embeddings.py      # Creates and stores embeddings in FAISS index
├── requirements.txt         # Required Python packages
├── templates/
│   └── index.html           # HTML template for the chat interface
├── tweets_metadata.json     # Metadata for the tweets
├── tweets.faiss             # FAISS index for embeddings
├── elon_musk_tweets.csv     # Raw tweet data
├── elon_musk_cleaned_tweets.csv # Cleaned tweet data
└── tweet_embeddings.json    # JSON file with tweet embeddings
```

---

## **Setup Instructions**

### **1. Prerequisites**
- Python 3.7 or above
- pip (Python package manager)

### **2. Clone the Repository**
```bash
git clone https://github.com/xdcai-creator/prototype.git
cd prototype
```

### **3. Create and Activate a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate    # For Windows
```

### **4. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **5. Preprocess Tweets**
Clean the raw tweet data:
```bash
python preprocess_tweets.py
```

### **6. Generate Embeddings**
Generate embeddings for the cleaned tweets:
```bash
python store_embeddings.py
```

---

## **Running the Application**

### **1. Start the Flask App**
```bash
python app.py
```

### **2. Access the Application**
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### **3. Chat Interface**
- Enter your query in the chat box and press **Submit**.
- The application retrieves relevant tweets and generates a response.

---

## **Implementation Details**

### **Preprocessing Tweets**
- The `preprocess_tweets.py` script removes noise such as URLs, hashtags, and unnecessary symbols from the tweets.

### **Embedding Generation**
- The `store_embeddings.py` script uses the SentenceTransformer model (`all-MiniLM-L6-v2`) to convert tweets into vector embeddings.
- Embeddings are stored in a FAISS index for efficient similarity search.

### **Query Handling**
- User queries are embedded and matched against the FAISS index using `query_embeddings.py`.
- The top 5 most similar tweets are retrieved for context.

### **Response Generation**
- The `generate_response.py` module uses GPT-Neo (via Hugging Face) to generate conversational responses based on the query and retrieved tweets.
- Parameters like `temperature` and `repetition_penalty` ensure meaningful and coherent responses.

### **Flask Application**
- `app.py` serves the chat interface, handling user input and displaying generated responses in a user-friendly format.

---

## **Technologies Used**
- **Flask**: Web framework for building the application.
- **FAISS**: Library for efficient similarity search.
- **Hugging Face Transformers**: For integrating the GPT-Neo language model.
- **SentenceTransformers**: For generating embeddings from tweets.
- **HTML/Jinja**: For creating the front-end interface.

---

## **How It Works**
1. **User Input**: The user submits a question via the chat interface.
2. **Tweet Retrieval**: FAISS retrieves the most relevant tweets based on the query.
3. **Response Generation**: GPT-Neo generates a response using the retrieved tweets as context.
4. **Display**: The application displays the question, relevant tweets, and the generated response.

---

## **Example Queries**
1. **What do you think about Mars?**
2. **What’s your opinion on Bitcoin?**
3. **How do you feel about AI and its future?**
4. **What’s your take on Dogecoin?**
5. **How do you approach innovation?**

---

## **Contributing**
Feel free to fork the repository and submit pull requests for improvements or new features.

---

## **License**
This project is licensed under the MIT License.

---
