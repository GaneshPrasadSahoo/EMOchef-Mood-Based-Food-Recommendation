# 🍽️ EMOchef – Mood Based Food Recommendation

EMOchef is an AI-powered application that suggests Indian food based on your **emotions**, detected either from a **face image** or by entering your **current mood**. It uses a deep learning model for facial emotion recognition and integrates a large language model (LLM) to recommend emotionally uplifting Indian dishes.

---

## 🌟 Features

- 🎭 Detects emotions from face images (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised)
- ✍️ Accepts manual mood input (e.g., "Sad", "Happy")
- 🍛 Recommends Indian dishes (savory, sweet, and light snack)
- 📜 Gives a short reason and a quick making process for each dish
- 💬 Built using Streamlit for an interactive UI
- 🤖 Powered by TensorFlow (ResNet model) + LangChain + Ollama (LLaMA 3)

---

## 📸 Emotion Classes

The model classifies emotions into the following 7 categories:

- Angry  
- Disgusted  
- Fearful  
- Happy  
- Neutral  
- Sad  
- Surprised  

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/GaneshPrasadSahoo/EMOchef-Mood-Based-Food-Recommendation.git
cd EMOchef-Mood-Based-Food-Recommendation

## Install the dependencies:
pip install -r requirements.txt

## Make sure you have Ollama and the llama3 model installed:
ollama run llama3

##🚀 Run the App
streamlit run final.py
