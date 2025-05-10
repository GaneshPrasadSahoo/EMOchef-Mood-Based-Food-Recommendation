import os 
import streamlit as st  
from langchain_ollama.llms import OllamaLLM

# Initialize the model
llm = OllamaLLM(model="llama3:latest")

# Set up the Streamlit page
st.set_page_config(page_title="Mood Based Food Recommendation")
st.title("EMOchef")
st.write("Welcome to EMOchef – Get Indian food suggestions based on how you feel!")

# Get user mood input
mood = st.text_input("How are you feeling today?", "")

if st.button("Processed"):
    if mood.strip() == "":
        st.warning("Please enter your current mood.")
    else:
        # Modified prompt
        prompt = (
            f"You are a professional Indian chef. Suggest a list of at least 3 Indian dishes "
            f"that are perfect for someone who is feeling {mood}. "
            f"The suggestions should be comforting, emotionally uplifting, and diverse (e.g., one savory, one sweet, one light snack). "
            f"Include a short reason for each recommendation."
        )
        
        response = llm(prompt)

        st.subheader("Here's what you might enjoy:")
        st.write(response)
