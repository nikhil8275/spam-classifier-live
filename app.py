import os
import pickle
import string
from pathlib import Path
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing required package 'nltk'. Activate your virtual environment "
        "or install it with: pip install nltk"
    ) from e
from flask import Flask, render_template, request

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

ps = PorterStemmer()

BASE_DIR = Path(__file__).resolve().parent

VECTORIZER_PATH = BASE_DIR / 'vectorizer.pkl'
MODEL_PATH = BASE_DIR / 'model.pkl'


def preprocessing_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


try:
    with open(VECTORIZER_PATH, 'rb') as vector_file:
        tfidf = pickle.load(vector_file)
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    tfidf = None
    model = None


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    confidence = None
    error = None
    model_loaded = tfidf is not None and model is not None

    if request.method == 'POST':
        input_sms = request.form['sms']
        if model_loaded:
            transformed_sms = preprocessing_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            confidence = model.predict_proba(vector_input)[0].max() * 100
            prediction = "Spam" if result == 1 else "Not Spam"
        else:
            error = (
                "Model files not found. "
                "Run `train_model.py` to generate `model.pkl` and `vectorizer.pkl`, "
                "then restart the app."
            )
    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        error=error,
        model_loaded=model_loaded,
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() in ['1', 'true', 'yes']
    app.run(host='0.0.0.0', port=port, debug=debug)

