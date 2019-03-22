#Importing libraries

import numpy as n
import matplotlib.pyplot as plt
import pandas as p
import pickle

#Importing Dataset

ds = p.read_csv('Salary_Data.csv')
x = ds.iloc[:, :-1].values
y = ds.iloc[:, 1].values

#Splitting the Dataset

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 1/3, random_state = 0)

#SPLR Fitting

from sklearn.linear_model import LinearRegression
Reg = LinearRegression()
Reg.fit(xtrain, ytrain)

#Result Prediction

predY = Reg.predict(xtest)

#Visualizing The Results(Training Set)

plt.scatter(xtrain, ytrain, color = 'red')
plt.plot(xtrain, Reg.predict(xtrain), color = 'blue')
plt.title('Salary vs Years Of Experience(Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing The Results(Test Set)

plt.scatter(xtest, ytest, color = 'red')
plt.plot(xtrain, Reg.predict(xtrain), color = 'blue')
plt.title('Salary vs Years Of Experience(Test Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

#Creating API

filename = 'SPLR_New.pkl'
pickle.dump(Reg, open(filename, 'wb'))

