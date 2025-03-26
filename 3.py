import pandas as pd
import math   #for dataframe/excel sheet operation
import numpy as np #for matrix operations
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("bcancer.csv")#performing EDA on dataset  #load the dataset
data=data.drop(["id",'Unnamed: 32'],axis=1)

data["diagnosis"]=data["diagnosis"].map({"M":1,"B":0})

print(data.head()) #printing first 5 rows of dataframe
print(data.tail()) #printing last 5 rows of dataframe
print(data.columns[data.isna().any()])#checking for NaN values in the dataframe
print(data.info()) #general information on dataset
print(data.describe()) #statistical description of the dataset

data=data.dropna()#drop rows that have NaN values
print(data.columns[data.isna().any()])#check again fro Nan values

#segregating data into input and output
x=data.iloc[:,1:].values #choosing input columns
y=data.iloc[:,0].values #choosing output columns

print(x)  #printing input columns
print(y)  #printing output columns


#splitting the dataset into training and testing the partions
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(x.shape)#x.shape=x_train.shape + x_test.shape

#Taining the model
from sklearn.linear_model import LogisticRegression #importing package
logistic_regressor = LogisticRegression(max_iter=10000) #intialising the algorithm
logistic_regressor.fit(x_train,y_train) #trainig the algo on training #dataset

#Evaluating the model
#using th trained model to predict on test data/testing the model
logistic_regressor_test_predictor = logistic_regressor.predict(x_test)

#getting the scores for model
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
print("accuracy : ",accuracy_score(y_test,logistic_regressor_test_predictor))
print("precision : ",precision_score(y_test,logistic_regressor_test_predictor))
print("recall : ",recall_score(y_test,logistic_regressor_test_predictor))
print("f1 score : ",f1_score(y_test,logistic_regressor_test_predictor))

# Step 5: Calculate the confusion matrix
cm = confusion_matrix(y_test, logistic_regressor_test_predictor)
# Step 7: Calculate the correlation matrix
correlation_matrix = data.corr()

# Step 6: Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.show()


# Step 8: Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()