from pathlib import Path
import pandas as pd
import numpy as np
import string
import pickle
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing required package 'nltk'. Activate your virtual environment "
        "or install it with: pip install nltk"
    ) from e
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

BASE_DIR = Path(__file__).resolve().parent
SPAM_CSV_PATH = BASE_DIR / 'spam.csv'
VECTORIZER_PATH = BASE_DIR / 'vectorizer.pkl'
MODEL_PATH = BASE_DIR / 'model.pkl'

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

print("NLTK data downloaded.")

# Load and clean data
spam = pd.read_csv(SPAM_CSV_PATH, encoding='latin1')
spam.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
spam.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
encoder = LabelEncoder()
spam['target'] = encoder.fit_transform(spam['target'])
spam = spam.drop_duplicates(keep='first')
print("Data loaded and cleaned. Shape:", spam.shape)

# Preprocessing function (matches app.py)
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

spam['transformed_text'] = spam['text'].astype(str).apply(preprocessing_text)

# Vectorize and train
tfidf = TfidfVectorizer(max_features=300)
X = tfidf.fit_transform(spam['transformed_text']).toarray()
Y = spam['target'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

mnb = MultinomialNB()
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, random_state=2)
mnb.fit(X_train, Y_train)
mlp.fit(X_train, Y_train)

# Save MLP as primary model (overwrites any old files)
with open(VECTORIZER_PATH, 'wb') as vector_file:
    pickle.dump(tfidf, vector_file)
with open(MODEL_PATH, 'wb') as model_file:
    pickle.dump(mlp, model_file)

print("Neural Network (MLP) model.pkl and vectorizer.pkl saved successfully.")
print("MLP Test accuracy:", mlp.score(X_test, Y_test))

