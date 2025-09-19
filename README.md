# Emoji Mood Classifier from Text

This project predicts the mood of a given text and suggests a relevant emoji using machine learning models.

## Project Structure

- `app/streamlit.py`: Streamlit web app for user interaction
- `data/dataset.csv`: Dataset used for training and evaluation
- `models/`: Saved models and vectorizers
- `src/`: Source code for preprocessing, training, and evaluation

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train models (optional, if not already trained):
   ```
   python -m src.train_lr
   python -m src.train_nb
   python -m src.train_svm
   ```
3. Start the Streamlit app:
   ```
   streamlit run app/streamlit.py
   ```

## Features
- Text preprocessing
- Multiple ML models (Logistic Regression, Naive Bayes, SVM)
- Emoji prediction based on text mood
- Interactive web interface

## Notes
- The `data/` folder is ignored by git for privacy.
- Model files are stored in `models/`.

## Author
Kandhavel-S
