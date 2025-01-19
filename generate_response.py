# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load GPT-Neo model and tokenizer
# model_name = "EleutherAI/gpt-neo-1.3B"  # Use a smaller model if required
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Assign the eos_token as the pad_token to resolve padding issues
# tokenizer.pad_token = tokenizer.eos_token

# def generate_response(context, query):
#     """
#     Generate a response using GPT-Neo based on the provided context and query.
    
#     Args:
#         context (str): Context text summarizing relevant information.
#         query (str): User query or question to be answered.

#     Returns:
#         str: Generated response from the model.
#     """
#     # Refined prompt with clear instructions
#     input_text = (
#         f"The following tweets summarize Elon Musk's thoughts:\n"
#         f"{context}\n\n"  # Include context
#         f"Using the tweets above, provide a clear, accurate, and concise response in 1-2 sentences.\n"
#         f"Question: {query}\n"
#         f"Answer:"
#     )

#     # Tokenize input with padding and truncation
#     inputs = tokenizer(
#         input_text,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512  # Ensure input doesn't exceed model limit
#     )

#     # Generate response with optimized settings
#     outputs = model.generate(
#         inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         max_new_tokens=50,  # Limit response length
#         temperature=0.7,  # Balance randomness
#         top_p=0.9,  # Nucleus sampling for diversity
#         repetition_penalty=2.0,  # Penalize repetition
#         pad_token_id=tokenizer.pad_token_id,
#         do_sample=True  # Enable sampling
#     )

#     # Decode the response and clean up text
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # Post-process to remove unwanted trailing text
#     response = response.split("Answer:")[-1].strip()
#     return response


from transformers import AutoModelForCausalLM, AutoTokenizer

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Assign the eos_token as the pad_token to resolve padding issues
tokenizer.pad_token = tokenizer.eos_token

def generate_response(context, query):
    """
    Generate a response using GPT-Neo based on the provided context and query.
    
    Args:
        context (str): Context text summarizing relevant information.
        query (str): User query or question to be answered.

    Returns:
        str: Generated response from the model.
    """
    # Refined prompt with explicit instructions
    input_text = (
        f"The following tweets summarize Elon Musk's thoughts:\n"
        f"{context}\n\n"
        f"Using these tweets, provide a clear, accurate, and concise answer to the question below in 1-2 sentences.\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    # Tokenize input with padding and truncation
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Generate response with optimized settings
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        temperature=0.5,  # More deterministic
        top_p=0.9,
        repetition_penalty=4.0,  # Stronger penalty for repetition
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True
    )

    # Decode and clean the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Answer:")[-1].strip()

    # Post-process: Ensure concise, non-repetitive response
    sentences = response.split('.')
    unique_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        stripped = sentence.strip()
        if stripped and stripped not in seen_sentences:
            unique_sentences.append(stripped)
            seen_sentences.add(stripped)

    # Return only the first 2 unique sentences
    return '. '.join(unique_sentences[:2]) + '.'
