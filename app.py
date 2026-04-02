import os
import pickle
import string
from pathlib import Path

# ✅ Define BASE_DIR FIRST
BASE_DIR = Path(__file__).resolve().parent

# ✅ Setup NLTK data path BEFORE using stopwords
import nltk
nltk_data_path = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, render_template, request

# ✅ Initialize tools AFTER download
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# ✅ Correct paths (same folder as app.py)
VECTORIZER_PATH = BASE_DIR / 'vectorizer.pkl'
MODEL_PATH = BASE_DIR / 'model.pkl'


# ✅ Optimized preprocessing
def preprocessing_text(text):
    text = text.lower()

    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()

    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)


# ✅ Load model safely
try:
    with open(VECTORIZER_PATH, 'rb') as f:
        tfidf = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
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
        try:
            input_sms = request.form.get('sms', '')

            if not input_sms.strip():
                error = "Please enter a message"

            elif model_loaded:
                transformed_sms = preprocessing_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])

                result = model.predict(vector_input)[0]
                confidence = model.predict_proba(vector_input)[0].max() * 100

                prediction = "Spam" if result == 1 else "Not Spam"

            else:
                error = "Model not loaded"

        except Exception as e:
            print("ERROR:", str(e))  # check in Render logs
            error = str(e)

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        error=error,
        model_loaded=model_loaded,
    )


if __name__ == '__main__':
    app.run(debug=True)