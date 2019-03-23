# Import dependencies
import pandas as pd
import numpy as np

df = pd.read_csv('Salary_Data.csv')


x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

#Splitting the Dataset

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 1/3, random_state = 0)

#SPLR Fitting

from sklearn.linear_model import LinearRegression
Reg = LinearRegression()
Reg.fit(xtrain, ytrain)

# Save your model
from sklearn.externals import joblib
joblib.dump(Reg, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
reg = joblib.load('model.pkl')

#Result Prediction

predY = reg.predict(xtest)
print(xtest)
print(predY)
