import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
def loadModels(model_path, encoder_path, verbose=False):
    model_path = os.path.join(model_path, "fake_news_model.h5")
    encoder_path = os.path.join(encoder_path, "tokenizer.h5")
    model = load_model(model_path)
    with open(encoder_path, 'rb') as pickle_file:
        encoder = pickle.load(pickle_file)
    return model, encoder

def transform(X, maxlen, verbose=False):
    tmp = np.array(X)
    tmp = tmp.reshape(1, tmp.shape[0])
    X = encoder.texts_to_sequences(tmp.tolist())
    return pad_sequences(X, maxlen)

def predict_news(txt, maxlen, clf_model, txt_encoder, verbose=False):
    X = preprocessing(txt, verbose)
    X = transform(X, maxlen, verbose)
    y = clf_model.predict(X)
    if y>0.5:
        return "True"
    else:
        return "False"

@app.route('/')
def home():
    return render_template("index.html")


@app.route("/predict",methods = ['POST'])
def predict():
    model, encoder = loadModels('models', 'models')
    news = [x for x in request.form.values()]
    prediction = predict_news(news, 700, model, encoder)

    return render_template("temp.html",prediction_text = 'News is $ {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug = True)