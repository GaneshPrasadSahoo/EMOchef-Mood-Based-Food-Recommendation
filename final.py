# It include both Deep Learning model with Text Based Mood Based Food Recommendation

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from langchain_ollama.llms import OllamaLLM

# Set up the page
st.set_page_config(page_title="EMOchef – Mood Based Food Recommendation", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main-title {
            font-size:40px !important;
            color: #FF5733;
            font-weight: bold;
        }
        .sub-title {
            font-size:24px !important;
            color: #1F618D;
            margin-bottom: 20px;
        }
        .emotion-title {
            color: #117A65;
            font-size: 22px !important;
        }
        .food-title {
            color: #884EA0;
            font-size: 22px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🍽️ EMOchef</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Get Indian food suggestions based on your mood – detected from an image or entered directly!</div>', unsafe_allow_html=True)

# Load your trained emotion model
model = tf.keras.models.load_model("E:/Emochef/project.h5")

# Define class labels
class_names = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# LLM Setup
llm = OllamaLLM(model="llama3:latest")

# Function to preprocess image for ResNet
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 224, 224, 3)
    return image_array

# Section: Upload image
st.markdown("### 📷 Upload a Face Image")
uploaded_file = st.file_uploader("Choose a face image (jpg/jpeg/png):", type=["jpg", "jpeg", "png"])

# Section: OR enter mood manually
st.markdown("### ✍️ Or Type How You're Feeling")
mood_input = st.text_input("Enter your mood manually (e.g., Happy, Sad, etc.)")

# Process Button
if st.button("🔍 Suggest Food"):
    mood = ""

    # Handle image input
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Detecting emotion..."):
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            predicted_class = class_names[np.argmax(predictions)]
        st.markdown(f'<div class="emotion-title">Detected Emotion: <b>{predicted_class}</b></div>', unsafe_allow_html=True)
        mood = predicted_class

    # Handle manual mood input if image not provided
    elif mood_input.strip() != "":
        mood = mood_input.strip().capitalize()
        st.markdown(f'<div class="emotion-title">Entered Mood: <b>{mood}</b></div>', unsafe_allow_html=True)

    else:
        st.warning("Please upload an image or enter your mood.")
    
    if mood:
        prompt = (
            f"You are a professional Indian chef. Suggest at least 3 Indian dishes suitable for someone feeling {mood}. "
            f"The suggestions should include a comforting dish, a sweet item, and a light snack. "
            f"Also include a short reason for recommending each dish and optionally a quick making process."
        )

        with st.spinner("Fetching food suggestions..."):
            response = llm(prompt)

        st.markdown('<div class="food-title">🍛 Food Recommendations:</div>', unsafe_allow_html=True)
        st.write(response)
