import numpy as np
import requests
from flask import Flask, jsonify
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

app = Flask(__name__)

# Load the trained model
model = load_model('my_cnn_model.h5')


# Define the API endpoint for prediction
@app.route('/predict', methods=['GET','POST'])
def predict():

    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'crude oil price Changes',
        'sortBy': 'publishedAt',
        'apiKey': 'f099ad81f7434501bbb8e1441347c466'  # Replace with your NewsAPI key
    }

    # Make API request and parse response
    response = requests.get(url, params=params)
    data = response.json()

    # Retrieve the news text from the API response
    news_text = data['articles'][0]['description']

    # Preprocess the news text
    tokenizer = Tokenizer(num_words=10000)  # define the tokenizer with a maximum number of words
    tokenizer.fit_on_texts([news_text])  # fit the tokenizer on the news text
    sequences = tokenizer.texts_to_sequences([news_text])  # convert the text to sequences of integers
    padded_sequences = pad_sequences(sequences, maxlen=100)  # pad the sequences to a fixed length

    # Make a prediction using the trained model
    prediction = model.predict(padded_sequences)[0]
    predicted_class = np.argmax(prediction)

    # Return the prediction as a JSON response
    prediction_list = prediction.tolist()
    return jsonify({'prediction': prediction_list})


if __name__ == '_main_':
    app.run(debug=False)