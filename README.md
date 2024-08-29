# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
### Program to implement the simple linear regression model for predicting the marks scored.
### Developed by: Kanagavel A K
### RegisterNumber: 212223230096
```py
/*

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
*/
```

## Output:
### Dataset
![image](https://github.com/user-attachments/assets/7b3d1ca8-22c5-4a51-b0f2-632d90002644)

### Head Values
![image](https://github.com/user-attachments/assets/4aabee05-2ccf-4549-bd23-58d47a802462)


### Tail Values
![image](https://github.com/user-attachments/assets/bc2f6b2d-e9fe-4782-b238-d2af5a6582f0)


### X values
![image](https://github.com/user-attachments/assets/372f87f0-b4fd-4da1-9bc0-24070cabcaf7)

### Y values
![image](https://github.com/user-attachments/assets/d6b6a40a-879c-488a-8f9d-00a0ac3d6690)

### Value of Y prediction:
![image](https://github.com/user-attachments/assets/16307b01-80eb-4fdc-8c2e-85b355522612)


### MSE,MAE and RMSE
![image](https://github.com/user-attachments/assets/c5e43740-692f-46ef-aead-80f44ef34849)


### Training Set
![image](https://github.com/user-attachments/assets/9d0c1f39-4273-491d-96b5-72302cba348a)

### Testing Set
![image](https://github.com/user-attachments/assets/fe8ea2a7-fca7-4e47-9216-55b207aa479c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
