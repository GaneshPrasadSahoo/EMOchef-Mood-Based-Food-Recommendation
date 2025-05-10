import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model (adjust path)
model = tf.keras.models.load_model("E:\Emochef\project.h5")

# Define class labels
class_names = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Preprocessing function for ResNet-based models
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")  # Convert to RGB
    image = image.resize((224, 224))  # Resize for ResNet input
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize to [0, 1]
    image_array = image_array.reshape(1, 224, 224, 3)  # Reshape for model
    return image_array

# Streamlit UI
st.title("Emotion Detection from Face Image")
st.write("Upload a face image to predict the emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions)]

    st.success(f"Predicted Emotion: **{predicted_class}**")
