import pandas  as pd #for dataframe handling

data = pd.read_csv("diabetes.csv") #loading the dataset

print(data.columns[data.isna().any()])#checking for nan values

#seperate the dataset into input and output values

x = data.iloc[:,:-1].values#choosing input columns
y = data.iloc[:,-1].values#choosing oytput columns

#splitting datset into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(x.shape)

#training and testing of algorithm
from sklearn.linear_model import LogisticRegression #importing package
logistic_regressor = LogisticRegression(max_iter=1000) #calling the algorithm
logistic_regressor.fit(x_train,y_train)#training the algorithm

print("....training complete...")#message to convey training is complete

logistic_regressor_test_prediction = logistic_regressor.predict(x_test)

#getting accurascy,precision of the model
from sklearn.metrics import accuracy_score,precision_score
print("ACCURACY OF THE MODEL",accuracy_score(y_test,logistic_regressor_test_prediction))
print("PRECISION OF THE MODEL",precision_score(y_test,logistic_regressor_test_prediction))


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,logistic_regressor_test_prediction))