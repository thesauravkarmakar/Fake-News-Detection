import warnings

warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os
from flask import Flask, request, render_template

app = Flask(__name__)


def loadModels(model_path, encoder_path):
    model_path = os.path.join(model_path, "model.h5")
    encoder_path = os.path.join(encoder_path, "tokenizer.tk")
    model = load_model(model_path)
    with open(encoder_path, 'rb') as pickle_file:
        encoder = pickle.load(pickle_file)
    return model, encoder


def preprocessing(par):
    X = []
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    return tmp


def transform(X, maxlen,verbose=False):
    tmp = np.array(X)
    tmp = tmp.reshape(1, tmp.shape[0])
    X = encoder.texts_to_sequences(tmp.tolist())
    return pad_sequences(X, maxlen)


def predict_news(txt, maxlen, clf_model, verbose=False):
    X = preprocessing(txt)
    X = transform(X, maxlen, verbose)
    y = clf_model.predict(X)
    if y > 0.5:
        return "Real"
    else:
        return "Fake"


@app.route('/')
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    model, encoder = loadModels('model', 'model', verbose=False)
    news = [x for x in request.form.values()]
    prediction = predict_news(news, 700, model, verbose=False)

    return render_template("temp.html", prediction_text='News is $ {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
