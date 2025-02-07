import streamlit as st
import torch 
from transformers import pipeline

def load_and_infer(review):
    
    label_mapping = {
        'LABEL_0': 'negative',
        'LABEL_1': 'positive'
    }
    
    classifier = pipeline("text-classification", model= "fine_tuned_bert", device = 0 if torch.cuda.is_available() else -1)
    
    result = classifier(review)[0]
    
    return label_mapping[result['label']]

st.title("GPU-Powered IMDb Sentiment Analyzer with BERT")

st.text("Enter a movie review to determine its sentiment (positive or negative)")

review  = st.text_area("Enter your review")

if st.button("Analyze"):
    result = load_and_infer(review)
    st.write(result)
    



    
    
    
    
    