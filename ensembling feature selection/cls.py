import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import sklearn as sk

def data(df):
	df_array = df.values
	X = df_array[:,:-1]
	Y = df_array[:,-1]
	return X,Y


def accuracy(clf,X_train, X_test, y_train, y_test):
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    return accuracy_score(y_test, y_predict)

def model(dataset, value):
	df =  pd.read_csv(dataset)
	X,Y = data(df)
	if value == 'Logistic Regression':
		clf1 = sk.linear_model.LogisticRegression()
	elif value == 'Naive Bayes':
		clf1 = sk.naive_bayes.GaussianNB() 
	elif value == 'Random Forest':
		clf1 = RandomForestClassifier(n_estimators=100)
	else:
		clf1 = sk.svm.LinearSVC(penalty="l1",dual=False)
 
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
	
	acc1 = accuracy(clf1,X_train, X_test, y_train, y_test)
	print("Accuracy without feature selection: %.2f"%(acc1))
	return acc1