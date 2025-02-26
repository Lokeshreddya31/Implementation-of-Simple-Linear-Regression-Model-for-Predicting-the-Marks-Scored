# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries. 
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: LOKESH REDDY A
RegisterNumber: 212223040104
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
#### DATASET:
![image](https://github.com/user-attachments/assets/d96a29ed-e5f4-467f-b497-1c4cadfd93a1)


#### HEAD VALUES:
![image](https://github.com/user-attachments/assets/03051e38-3be3-4a05-a2f3-25dc7bbee4d2)


#### TAIL VALUES:
![image](https://github.com/user-attachments/assets/86a92e9a-3de9-4425-9854-542a5ffacf5b)


#### X AND Y VALUES:
![image](https://github.com/user-attachments/assets/863ee89a-6ad6-418f-b59d-a071b92bead6)


#### PREDICTION VALUES:
![image](https://github.com/user-attachments/assets/d77454b8-6781-4357-ba0e-aa250601f850)


#### TRAINING SET:
![image](https://github.com/user-attachments/assets/49bb7e56-bfc5-4b06-939e-56c9b749d358)


#### TESTING SET:
![image](https://github.com/user-attachments/assets/0c6ae6e7-4ad3-4714-82ca-422fba9a9e43)


#### MSE,MAE AND RMSE:
![image](https://github.com/user-attachments/assets/299631ba-e579-4c98-84b5-4f4398734152)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
