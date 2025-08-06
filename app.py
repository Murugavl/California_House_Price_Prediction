import pickle
from flask import Flask, request, jsonify,url_for, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)
## load the model
model = pickle.load(open('XGBoost_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    req_json = request.get_json()

    if 'data' not in req_json:
        return jsonify({'error': 'Missing "data" key in JSON'}), 400

    data = req_json['data']

    try:
        print(data)
        print(np.array(list(data.values())).reshape(1, -1))
        new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
        output = model.predict(new_data)
        print(output[0])
        return jsonify(float(output[0]))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)