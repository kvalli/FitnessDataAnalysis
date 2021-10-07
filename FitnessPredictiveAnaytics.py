import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#read data from csv
df = pd.read_csv("FitnessData.csv")
test_df = pd.read_csv("FitnessTestData.csv")
print(df.columns)

#Drop columns that are not necessary or irrelevant for model building
df = df.drop(['Name', 'Age', 'Device_Id', 'Timestamp','Temperature'],axis = 1)
test_df = test_df.drop(['Name', 'Age', 'Device_Id', 'Timestamp','Temperature'],axis = 1)

X = df.drop(['Ideal_weight'],axis =1)
y = df['Ideal_weight']

#Drop columns unnecessary for Test Data
test_data = test_df.drop(['Ideal_weight'],axis =1)
y_test_actual = test_df['Ideal_weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#scale the data uniformly
scaler =  StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# we are selecting Linear Regression model as we have to predict ideal weight of the persons
reg = LinearRegression(normalize=True).fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("R2 score for model",r2_score(y_test,y_pred))

test_data = scaler.transform(test_data)
y_pred_new = reg.predict(test_data)
print("R2 score for test data",r2_score(y_test_actual,y_pred_new))