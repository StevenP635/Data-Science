# -*- coding: utf-8 -*-
"""
HW6 code
@author: Steven Paredes
@abc123: obr635
@Date: 
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import statistics
import pandas as pd
#%% Question 1 setup
data = pd.read_excel('hw6q1.xlsx', sheet_name = 'hw5q1')
#%% Question 1A
print("Question 1A data shape: ", data.shape)
plt.figure()
plt.title('Fig 1A raw data box plot')
#plt.boxplot(data) something is going wonky here
data.boxplot()
plt.show()

#%% Question 1B
datalg2 = np.log2(data)
plt.figure()
plt.title('Fig 1B log2 transform')
datalg2.boxplot()
plt.show()

#%% Question 1C
desc = data.describe()
print("\nQuestion 1c answer:")
print(desc)

#%% Question 1D
plt.figure()
plt.title('Fig 1D')
data.hist(density = True)
plt.show()

#%% Question 1E
plt.figure()
plt.title('Fig 1E')
datalg2.hist(density = True)
plt.show()

#%% Question 1F
print("\nQuestion 1F answer:")
print("It would seem that A has a normal distribution, B has an exponential distribution,")
print("c has a log normal distribution, and d has a pareto distribution.\n")

#%% Question 2A
data2 = pd.read_csv('brfss.csv',header=None).values
colName = data2[0]
data2 = pd.DataFrame(data2)
data2.columns = colName
data2 = pd.DataFrame(data2.drop('wtkg2',axis=1).dropna(axis=0, how='any').values)
colName = np.delete(colName, 4)
data2.columns = colName
data2 = pd.DataFrame(data2.drop('wtyrago',axis=1).values)
colName = np.delete(colName, 3)
data2.columns = colName
data2['age'] = data2['age'].astype(float)
data2['weight2'] = data2['weight2'].astype(float)
data2['htm3'] = data2['htm3'].astype(float)
data2 = data2.drop(data2.columns[0],axis=1).values
colName = np.delete(colName, 0)
data2[:,3][data2[:,3] == '1'] = True
data2[:,3][data2[:,3] == '2'] = False
data2[:,3][data2[:,3] == 1] = True
data2[:,3][data2[:,3] == 2] = False
brfss = pd.DataFrame(data2)
brfss.columns = colName

brfssMale = pd.DataFrame(data2[data2[:,3] == True])
brfssMale.columns = colName

brfssFemale = data2[data2[:,3] == False]
brfssFemalelt20 = brfssFemale[brfssFemale[:,0] <= 20]
brfssFemalelt20 = pd.DataFrame(brfssFemalelt20)
brfssFemalelt20.columns = colName
brfssFemale = pd.DataFrame(brfssFemale)
brfssFemale.columns = colName
print("\nQuestion 2A\nbrfss DataFrame shape:", brfss.shape)

#%% Question 2B
print("\nQuestion 2B Answers")
print("brfss DataFrame max age: ", brfss['age'].max())
print("brfss DataFrame mean weight: ", brfss['weight2'].mean())
print("brfss DataFrame mean weight of males: ", brfss.loc[brfss['sex'] == True ,'weight2'].mean())
print("brfss DataFrame mean height of females: ", brfss.loc[brfss['sex'] == False ,'htm3'].mean())
print("brfss DataFrame mean weight for female younger than 20 years old: ", brfss.loc[(brfss['sex'] == False) & (brfss['age'] < 20) ,'weight2'].mean())
print("brfss DataFrame number of males in the dataset: ", len(brfss.loc[brfss['sex'] == True]))
print("brfss DataFrame number of individuals in the dataset height > 190cm and weight < 50kg: ", len(brfss.loc[(brfss['htm3'] > 190) & (brfss['weight2'] < 50) ,'weight2']))
print("brfss DataFrame average height of females whose weight is between 59 and 61 kg: ", brfss.loc[(brfss['sex'] == False) & (brfss['weight2'] <= 61) & (brfss['weight2'] >= 59) ,'htm3'].mean())
print("brfss DataFrame row 2001 to row 2010: \n", brfss.loc[2001:2011])
print("\nbrfss DataFrame rows with row index from 2001 to 2010: \n", brfss.iloc[2001:2011])

#%% Question 3A
"""
a. (10 pts) From the brfss DataFrame created in Q2a, use the height column as Y, and weight column as X
to perform a simple linear regression (See 6-regression.ppt slide #23, lr.fit). Print out the equation that is
obtained by the linear regression in the form of “height = a * weight + b” (replace a and b with the 
values obtained from the linear regression, lr.coef_ and lr.intercept_.) (Note: you need to either make X
a DataFrame or reshape X to be a n x 1 numpy array using reshape(-1, 1); see simpleRegression.ipynb.)
"""
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from pandas import Series

height = np.array(brfss['htm3'])
weight = np.array(brfss['weight2']).reshape(-1,1)
lr = linear_model.LinearRegression()
lr.fit(weight, height)
coef = Series(lr.coef_)
intercept = Series(lr.intercept_)
print("\nQuestion 3A:")
print("height = ", coef[0], " * weight + ", intercept[0])

#%% Question 3B
"""
b. (4 pts) Use the linear regression object that you obtained in 3a to predict the height of an individual
whose weight is 60kg (with lr.predict). Print out the predicted height.
"""
zeta = np.array([60]).reshape(-1,1)
pred = lr.predict(zeta)
print("\nQuestion 3B:")
print("Predicted height of an individual whose weight is 60kg: ", pred)

#%% Question 3C
#c. (6 pts) Compute the MSE and R-square of the linear regression. (Slide #23, Imported as r2 and mse.)
pred = lr.predict(weight)
print('\nQuestion 3c:')
print('r2 = %.3f' % r2(height, pred))
print('mse = %.3f' % mse(height, pred))

#%% Question 3D
"""
d. (10 pts) From the brfss DataFrame above, use the height column as Y, and two columns, weight and
male, as X, to perform a multiple linear regression (slides #34). Print out the equation that is obtained by
the linear regression in the form of “height = a * weight + b * male + c” (replace a, b and c with the
values obtained from the linear regression.)
"""
y = brfss['htm3']
x = brfss[['weight2','sex']]
lr2 = linear_model.LinearRegression()
lr2.fit(x,y)
coef2 = Series(lr2.coef_)
intercept2 = Series(lr2.intercept_, index = ['weight2','sex'])
print("\nQuestion 3d:")
print("height = ", coef2[0], " * weight + ", coef2[1] , " * male + ", intercept2[0])

#%% Question 3E
"""
e. (3 pts) Use the linear regression object that you obtained in 3d to predict the height of a male whose
weight is 60kg. Print out the predicted height.
"""
zeta = np.array([60, True]).reshape(1,2)
pred2 = lr2.predict(zeta)
print("\nQuestion 3e:")
print("Predicted height of a male whose weight is 60kg: ", pred2)

#%% Question 3F
"""
f. (3 pts) Use the linear regression object that you obtained in 3d to predict the height of a female whose
weight is 60kg. Print out the predicted height. (Just FYI, how does it compare with the value in Q2b.viii?)
"""
zeta = np.array([60, False]).reshape(1,2)
pred2 = lr2.predict(zeta)
print("\nQuestion 3f:")
print("Predicted height of a female whose weight is 60kg: ", pred2)

#%% Question 3G
#g. (4 pts) Compute the MSE and R-square of the linear regression
pred2 = lr2.predict(x)
print("\nQuestion 3g:")
print('r2 = %.3f' % r2(y, pred2))
print('mse = %.3f' % mse(y, pred2))

#%% Question 4A
"""
4. Multiple linear regression (30 points, optional)
a. (5 pts) Load data stored in HDF5 format into python using the following statement: hdfstore =
pd.HDFStore('hw6q4.h5'). Perform a least square multiple linear regression between the objects x and y
in hdfstore (hdfstore[‘x’] and hdfstore[‘y’]). Report the R-squared and Mean Square Error (MSE) of the
regression. Plot the coefficients in a bar chart.
"""
hdfstore = pd.HDFStore('hw6q4.h5')
datax = pd.DataFrame(hdfstore['x'])
datay = hdfstore['y']
lr3 = linear_model.LinearRegression()
lr3.fit(datax,datay)
pred4 = lr3.predict(datax)
print("\nQuestion 4a:")
print('r2 = %.3f' % r2(datay, pred4))
print('mse = %.3f' % mse(datay, pred4))

#%% Question 4B
"""
b. (10 pts) Perform bootstrap to estimate the standard error of the coefficients obtained in 3a, and
calculate the statistical significance (p-value) of each coefficient (the probability that the coefficient is
equal to zero). Plot the -log10(p-value) in a bar chart. (See example in slide #37 and #39.)
"""
n = 100
dataSize = len(datay)
coef_bs = np.zeros((n,datax.shape[1]))
#print(datay.feature_names)
for i in range(n):
    sample = np.random.choice(np.arange(dataSize), size=dataSize, replace=True)
    newY = datay[sample]
    newX = datax.iloc[sample, :]
    lr.fit(newX, newY)
    coef_bs[i,:] = lr.coef_
coef_bs = pd.DataFrame(coef_bs)
plt.figure()
plt.title('Figure 4B (sorry not in -log10 scale)')
coef_bs.boxplot(rot=90);
plt.show()
"""
        Ok for this I had no idea how to put the plot in -log10 scale I hope that this doesnt take too many points off
"""
#%% Question 4C
"""
c. (10 pts) Perform lasso regression between x and y using alpha = 2**i, for -6 < i <6. For each value of
alpha, compute the R-squared as well as the sum of coefficients. Plot the R-squared, MSE, and the sum
of absolute value of the coefficients against the alpha values, in three lines in the same graph. Based on
the graph, what is the recommended value(s) of alpha that you should use? What is the R2 and MSE of
the fit? Plot the coefficients resulted from the lasso regression with the alpha parameter you choose.
"""
#%% Question 4D
"""
d. (5 pts) Transform the x matrix by dividing each column with the scaling factor stored in the object
hdfstore[‘sf’], and then perform a multiple linear regression between the transformed x matrix and the y
vector. Report the R-squared and the Mean Square Error of the regression. Use a graph to compare the
coefficients from the regression with the expected coefficient stored in hdfstore[‘coef’].
"""