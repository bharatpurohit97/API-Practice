import numpy as n
import sklearn
import pickle
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify

def predreg(num1):
    output = {'output prediction of salary' : 0}
    x_input = n.array([num1]).reshape(1,2)
    filename = 'SPLR_New.pkl'
    m1 = pickle.load(open(filename, 'rb'))
    output['output prediction of salary'] = m1.predict(x_input)[0]
    print(output)
    return output

app = Flask(__name__)

@app.route("/")
def index():
    return 'Salary_Predict!!'

@app.route("/salary_prediction", methods = ['GET'])
def calc_Salary_Predict():
    body = request.get_data()
    #header = request.headers

    try:
        num1 = int(request.args['x1'])
        
        if (num1 != None):
            res = predreg(num1)
                        
        else:
            res = {'sucess' : False,
                   'message' : 'Input Data is not correct'}

    except:
        res = {'sucess' : False,
               'message' : 'Unknown Error'}
        
    return jsonify(res)


if __name__ == "__main__":
    app.run(debug = True, port = 8791)
        
