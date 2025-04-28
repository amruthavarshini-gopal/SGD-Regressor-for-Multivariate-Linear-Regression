# SGD-Regressor-for-Multivariate-Linear-Regression

## Aim:

To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

Step 1:Start the program.

Step 2: Load the California Housing dataset and select the first three features as inputs (X), and the target and an additional feature (Y) for prediction..

Step 3: Scale both the input features (X) and target variables (Y) using StandardScaler.

Step 4: Initialize SGDRegressor and use MultiOutputRegressor to handle multiple output variables.

Step 5: Initialize SGDRegressor and use MultiOutputRegressor to handle multiple output variables.

Step 6: Train the model using the scaled training data, and predict outputs on the test data.

Step 7: Inverse transform predictions and evaluate the model using the mean squared error (MSE). Print the MSE and sample predictions.

Step 6:Stop the program.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house
and number of occupants in the house with SGD regressor.
Developed by: Amruthavarshini Gopal
RegisterNumber: 212223230013 
*/
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
##Load the dataset California Housing
dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
df.head()
## first three features as input
x=df.drop(columns=['AveOccup','HousingPrice'])
## aveoccup and housingprice as output
y=df[['AveOccup','HousingPrice']]
## split the data into training and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
## scale the features and target variable
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
#initialize the SGDRegressor
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
#we need to use MultiOutputRegressor to handle multiple output variables
multi_output_sgd=MultiOutputRegressor(sgd)
#train the model
multi_output_sgd.fit(x_train,y_train)
#predict on the test data
y_pred=multi_output_sgd.predict(x_test)
#inverse transforms the predictions to get them back to the original scale
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
#evaluate the model using mse
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",y_pred[:5])
```

## Output:

### Head of the Dataset
![Screenshot 2025-03-10 155353](https://github.com/user-attachments/assets/cc9cc1a6-dc68-4bb4-b67f-27eaed1a18fe)

### Mean Squared Error and Predicted Values
![Screenshot 2025-04-28 144407](https://github.com/user-attachments/assets/e1f36f8f-5573-49f3-b589-83e95dcd5bfb)

## Result:

Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
