# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import json
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if reg:
        try:
            json_ = request.get_json
            data = pd.read_csv('api-test.csv')
            print (data)
            res = dict()
            prediction = reg.predict(data)
            for i in range(len(prediction)):
                res[i] = prediction[i]

            return jsonify(res)

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    reg = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')

    app.run(port=port, debug=True)
