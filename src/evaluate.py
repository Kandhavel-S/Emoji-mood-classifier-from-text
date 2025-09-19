import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from src.preprocess import clean_text, load_vectorizer

def evaluate(model_path="models/svm_model.pkl", dataset="data/dataset.csv"):
    df = pd.read_csv(dataset)
    df['clean'] = df['text'].apply(clean_text)
    X, y = df['clean'], df['emoji']

    vectorizer = load_vectorizer()
    X_tfidf = vectorizer.transform(X)

    model = joblib.load(model_path)
    preds = model.predict(X_tfidf)

    print("Accuracy:", accuracy_score(y, preds))
    print(classification_report(y, preds))

if __name__ == "__main__":
    evaluate()
