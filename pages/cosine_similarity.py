import streamlit as st
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_similarity(text1, text2):
    """Compute the cosine similarity between two texts."""
    # Encode the texts
    inputs_1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    inputs_2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
    
    # Generate embeddings
    with torch.no_grad():
        outputs_1 = model(**inputs_1)
        outputs_2 = model(**inputs_2)
        
        # Use the mean of the last hidden state as the sentence embedding
        embeddings_1 = outputs_1.last_hidden_state.mean(dim=1)
        embeddings_2 = outputs_2.last_hidden_state.mean(dim=1)

    # Normalize the embeddings
    embeddings_1 = torch.nn.functional.normalize(embeddings_1, dim=-1).numpy()  # Ensure correct dimension and convert to numpy
    embeddings_2 = torch.nn.functional.normalize(embeddings_2, dim=-1).numpy()  # Ensure correct dimension and convert to numpy
    
    # Compute cosine similarity (ensuring embeddings are 1-D arrays for scipy's cosine function)
    cosine_sim = 1 - cosine(embeddings_1[0], embeddings_2[0])
    
    return cosine_sim



# Streamlit application layout
st.title("Text Similarity Checker")

# User inputs
text1 = st.text_area("Enter first text:", "Type text here...")
text2 = st.text_area("Enter second text:", "Type text here...")

if st.button("Compare"):
    if text1 and text2:
        similarity_score = get_similarity(text1, text2)
        st.write(f"Similarity score: {similarity_score:.4f}")
    else:
        st.write("Please enter both texts to compare their similarity.")
