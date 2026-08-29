import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as linearRegression
from sklearn.preprocessing import PolynomialFeatures


df=pd.read_csv('Position_Salaries.csv')
X=df.iloc[:, 1].values
y=df.iloc[:, 2].values
poly_reg=PolynomialFeatures(degree=4)

X_poly=poly_reg.fit_transform(X.reshape(-1,1))
Linear_model=linearRegression.LinearRegression()
Linear_model.fit(X_poly,y)
plt.scatter(X,y,color='red')
plt.plot(X,Linear_model.predict(X_poly),color='blue')
plt.show()


