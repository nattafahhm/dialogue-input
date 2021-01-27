import numpy as np 
import pickle
from preprocessing import decontract, cleanPunc, keepAlpha, removeStopWords, stemming
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

import os
from flask import Flask, jsonify,request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def main():
    return "Hello World!"

@app.route("/predict",methods=["POST"])
def predict():
    text = request.form.get('text')
    
    model = tf.keras.models.load_model('en-model-GloVe-LSTM.h5')
    with open('en-tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
        
    seq = loaded_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=27, padding="post")
    pred = model.predict(padded)
    pred = pred.round()
    pred = pred.astype(int)
    pred = pred[0]
    pred = ["yes" if x==1 else "no" for x in pred]
    categories = ['Influenza', 'Diarrhea', 'Hayfever', 'Cough', 'Headache', 'Fever', 'Runnynose', 'Cold']
    labels = []

    for i, category in enumerate(categories):
        wordDicts = {}
        wordDicts[category] = pred[i]
        labels.append(wordDicts.copy())  

    return jsonify(labels)
    
if __name__ == "__main__":        
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
#     app.run(debug=True)