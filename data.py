
import pandas as pd
import numpy as np

from sklearn import preprocessing


def load_data_train():

	train_df, test_df = load_data()


	X_train_df = train_df.drop(['label'], axis=1)
	Y_train_df = train_df['label']

	# print(X_train_df[:2].to_string())
	# exit(0)

	############################## Scale ##############################
	# X_train = preprocessing.MinMaxScaler().fit_transform(X_train.values)

	X_train = X_train_df.values
	Y_train = Y_train_df.values

	############################## Split ##############################
	split = int(len(X_train) * 0.1)

	X_test = X_train[:split]
	X_train = X_train[split:]

	Y_test = Y_train[:split]
	Y_train = Y_train[split:]

	return (X_train, Y_train), (X_test, Y_test)

def load_data_predict():

	train_df, test_df = load_data()

	idStart = 1
	X_id = range(idStart, len(test_df) + idStart)
	label = ['ImageId', 'label']
	X_test_df = test_df.drop('label', axis=1)

	############################## Scale ##############################
	# X_train = preprocessing.MinMaxScaler().fit_transform(test_df.values)

	############################## Split ##############################

	X_test = X_test_df.values

	return (X_test, X_id, label)


def load_data():

	# le = preprocessing.LabelEncoder()
	# lb = preprocessing.LabelBinarizer()

	train_df = pd.read_csv('datasets/train.csv')
	test_df = pd.read_csv('datasets/test.csv')

	train_objs_num = len(train_df)
	datasetAll = pd.concat(objs=[train_df, test_df], axis=0)


	train = datasetAll[:train_objs_num]
	test = datasetAll[train_objs_num:]

	return(train, test)
