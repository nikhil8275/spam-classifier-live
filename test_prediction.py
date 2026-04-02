from pathlib import Path
import pickle
import string
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing required package 'nltk'. Activate your virtual environment "
        "or install it with: pip install nltk"
    ) from e

BASE_DIR = Path(__file__).resolve().parent
VECTORIZER_PATH = BASE_DIR / 'vectorizer.pkl'
MODEL_PATH = BASE_DIR / 'model.pkl'

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

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
    print("Model loaded successfully.")
    
    test_messages = [
        "Hello, how are you?",
        "Congratulations! You've won $1000. Call now!",
        "Free entry to win $5000. Reply STOP to end."
    ]
    
    for msg in test_messages:
        transformed_sms = preprocessing_text(msg)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        confidence = model.predict_proba(vector_input)[0].max() * 100
        pred = "Spam" if result == 1 else "Not Spam"
        print(f"Message: {msg}")
        print(f"Prediction: {pred} (confidence: {confidence:.2f}%)\\n")
except FileNotFoundError as e:
    print("Error:", e)

