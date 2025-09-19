import streamlit as st
import joblib
from src.preprocess import clean_text, load_vectorizer

# Load model & vectorizer
model = joblib.load("models/svm_model.pkl")   # change model here (nb_model.pkl / lr_model.pkl / svm_model.pkl)
vectorizer = load_vectorizer()

st.title("🤖 Emoji Mood Classifier")
st.write("Type a message and I'll predict the emoji!")

user_input = st.text_area("Your Message:", "")

if st.button("Predict"):
    if user_input.strip():
        clean = clean_text(user_input)
        vector = vectorizer.transform([clean])
        prediction = model.predict(vector)[0]
        st.markdown(f"### Predicted Emoji: {prediction}")
    else:
        st.warning("Please type a message before predicting.")
