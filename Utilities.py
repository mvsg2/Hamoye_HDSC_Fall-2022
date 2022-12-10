import pandas as pd; import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

def label_encode(dataframe, categorical_column:str):
	"""Returns the dataframe with the categorical variable(s) label-encoded"""
	en= LabelEncoder()
	dataframe[categorical_column] = en.fit_transform(dataframe[categorical_column])
	return dataframe

def standard_scale(dataframe):
	"""Implements the Z-score normalization criterion: (xi - mean(x1, x2, ...)) / standard_deviation(x1, x2, ...)"""
	sc = StandardScaler()
	dataframe_norm = sc.fit_transform(dataframe)
	dataframe_norm = pd.DataFrame(dataframe_norm)
	return dataframe_norm

def minmax_scale(dataframe):
	"""Implements the MinMax normalization criterion: (xi - mean(x1, x2, ...)) / max(x1, x2, ...)"""
	sc = MinMaxScaler()
	dataframe_norm = sc.fit_transform(dataframe)
	dataframe_norm = pd.DataFrame(dataframe_norm)
	return dataframe_norm

def print_cm(y_pred, y_test):
	cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
	print("The confusion matrix is:\n", cm)
	print()

def print_report(y_pred, y_test):
	cr = classification_report(y_pred=y_pred, y_true=y_test)
	print("================== CLASSIFICATION REPORT =======================")
	print(cr, end='')
	print("================================================================")

def print_accuracy(y_pred, y_test):
	accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
	print("Accuracy: {%.2f}%".format(100*accuracy))

def print_precision(y_pred, y_test):
	precision = precision_score(y_pred=y_pred, y_true=y_test)
	print("Precision:", precision)

def print_recall(y_pred, y_test):
	recall = recall_score(y_pred=y_pred, y_true=y_test)
	print("Reall:", recall)

def print_f1_score(y_pred, y_test):
	f1 = f1_score(y_pred=y_pred, y_true=y_test)
	print("F1-score:", f1)