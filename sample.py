import numpy as np
import pandas as pd

df=pd.read_csv('Salary_Data.csv')

x=df["YearsExperience"].values
y=df["Salary"].values

m=0
b=0
L=0.01
n=len(x)
for i in range(2000):
    y_pred=m*x+b
    error=y-y_pred
    dm=(-2/n)*sum(x*error)
    db=(-2/n)*sum(error)
    m=m-L*dm
    b=b-L*db
    if i%100==0:
        mse=np.mean((y-y_pred)**2)
        print("mse:",mse)
print("m:",m)
print("b:",b)
exp = 3

predicted_salary = m * exp + b

print(predicted_salary)

y_res= np.sum((y - y_pred)**2)
y_total= np.sum((y - np.mean(y))**2)
r_squared=1- (y_res/y_total)
print("R-squared:", r_squared)