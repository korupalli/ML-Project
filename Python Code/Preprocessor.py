import pandas as pd
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

def initial_preprocess(frame):
	frame['Timestamp'] = frame.Timestamp.apply(lambda x: x.split())
	frame['Day'] = frame.Timestamp.apply(lambda x: x[0])
	frame['Month'] = frame.Timestamp.apply(lambda x: x[1])
	frame['Date'] = frame.Timestamp.apply(lambda x: x[2])
	frame['Time'] = frame.Timestamp.apply(lambda x: x[3].split(':')[0])
	frame['Year'] = frame.Timestamp.apply(lambda x: x[5])
	
	frame.replace('null;', np.NaN,inplace=True)
	
	month_map={'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11 , 'Dec':12}
	frame.Month = frame.Month.map(month_map)

	day_map={'Mon':1, 'Tue':2, 'Wed':3, 'Thu':4, 'Fri':5, 'Sat':6, 'Sun':7}
	frame.Day = frame.Day.map(day_map)

	frame.Month = frame.Month.astype(np.object)
	frame.Day = frame.Day.astype(np.object)
	frame.Date = frame.Date.astype(np.object)
	frame.Time = frame.Time.astype(np.object)
	frame.Year = frame.Year.astype(np.object)

	frame.Mentions.replace(np.NaN,0,inplace=True)
	frame['Mentions_count'] = frame.Mentions.apply(lambda x:0 if x==0 else len(x.split()))

	frame.Hashtags.replace(np.NaN,0,inplace=True)
	frame['Hashtags_count'] = frame.Hashtags.apply(lambda x:0 if x==0 else len(x.split()))
	
	frame['Pos'] = frame.Sentiment.apply(lambda x: int(x.split()[0]))
	frame['Neg'] = frame.Sentiment.apply(lambda x: int(x.split()[1]))
	frame = frame.drop(columns=['Timestamp','Entities','Username','Sentiment'])
	
	return frame
	
def Mention_Hashtag_Freq(frame):
	arr = np.array(frame.Mentions[frame.Mentions != 0].str.split().apply(lambda x: np.array(x,dtype=object)))
	Mention = pd.Series(np.concatenate(arr)).value_counts()
	
	arr = np.array(frame.Hashtags[frame.Hashtags != 0].str.split().apply(lambda x: np.array(x,dtype=object)))
	Hashtag = pd.Series(np.concatenate(arr)).value_counts()
	
	return Mention, Hashtag

def Preprocess(frame):
	#Iniitial preprocessing 
	#selecting only retweets>10
	frame.drop(frame.index[frame[frame['Retweets'] <= 10].index.values], inplace=True)
	frame.reset_index(inplace=True)
	frame.drop(frame.columns[[0]], axis = 1, inplace = True)
	
	frame = initial_preprocess(frame)
	
	#mentions_count, Hashtags_count = Mention_Hashtag_Freq(frame)
	mentions_count = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Mentions_count.csv')
	#entions_count.set_index('Unnamed: 0', inplace=True)
	
	Hashtags_count = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Hashtags_count.csv')
	Hashtags_count.set_index('Unnamed: 0', inplace=True)
	
	#Dropping duplicated index columns
	#frame.drop(frame.columns[[0, 1]], axis = 1, inplace = True)
	
	#Binning the retweet column and applying LabelEncoder
	frame['bin'] = pd.cut(frame['Retweets'], [10, 50, 100, 1000, 10000, 275530])
	frame['bin'] = frame.bin.apply(preprocessing.LabelEncoder().fit_transform)
	
	y = frame.bin
	X = frame.drop(['bin','Retweets'], axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=27)
	
	X = pd.concat([X_train, y_train], axis=1)
	
	counts = y_train.value_counts().values
	
	#Reducing size of dataset
	X_ = pd.concat([X[X['bin'] == 0].sample(min(int(counts[0]/100), counts[0])), X[X['bin'] == 1].sample(min(int(counts[0]/100), counts[1])),X[X['bin'] == 2].sample(min(int(counts[0]/100), counts[2])),X[X['bin'] == 3].sample(min(int(counts[0]/100), counts[3])), X[X['bin'] == 4].sample(min(int(counts[0]/100), counts[4]))], axis=0)
	
	mentions_count.set_index('Unnamed: 0', inplace=True)
	Hashtags_count.set_index('Unnamed: 0', inplace=True)

	def funct1(string):
		if string == 0:
			return 0
		final = 0
		for i in string.split():
			final += mentions_count.loc[i].values[0]
		return final
		
	def funct2(string):
		if string == 0:
			return 0
		final = 0
		for i in string.split():
			final += mentions_count.loc[i].values[0]
		return final

	test = pd.concat([X_test, y_test], axis=1)
	
	X_['Mentions_score'] = X_['Mentions'].apply(funct1)
	X_['Hashtags_score'] = X_['Hashtags'].apply(funct2)
	test['Mentions_score'] = test['Mentions'].apply(funct1)
	test['Hashtags_score'] = test['Hashtags'].apply(funct2)
	
	X_['Mentions_score_avg'] = X_['Mentions_score']/X_['Mentions_count']
	X_.Mentions_score_avg.replace(np.NaN,0,inplace=True)

	X_['Hashtags_score_avg'] = X_['Hashtags_score']/X_['Hashtags_count']
	X_.Hashtags_score_avg.replace(np.NaN,0,inplace=True)

	test['Mentions_score_avg'] = test['Mentions_score']/test['Mentions_count']
	test.Mentions_score_avg.replace(np.NaN,0,inplace=True)

	test['Hashtags_score_avg'] = test['Hashtags_score']/test['Hashtags_count']
	test.Hashtags_score_avg.replace(np.NaN,0,inplace=True)
	
	return X_, test

def Undersampling(frame):
	y = frame.bin
	X = frame.drop(['bin','Retweets', 'Mentions', 'Hashtags'], axis=1)

	# setting up testing and training sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=27)
	
	#First lets see which classifier is workink good on our data
	
	#Checking with dummy classifier with strategy as most frequent
	dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
	dummy_pred = dummy.predict(X_test)
	# checking unique labels
	print('DummyClassifier')
	# checking accuracy
	print('Accuracy score f1_score recall_score', accuracy_score(y_test, dummy_pred), f1_score(y_test, dummy_pred, average = 'macro'), recall_score(y_test, dummy_pred, average = 'macro'))
	
	lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)
	lr_pred = lr.predict(X_test)
	print('Logistic Regression')
	print('Accuracy score f1_score recall_score', accuracy_score(y_test, lr_pred), f1_score(y_test, lr_pred, average = 'macro'), recall_score(y_test, lr_pred, average = 'macro'))
	
	rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
	rfc_pred = rfc.predict(X_test)
	print('Random Forest')
	print('Accuracy score f1_score recall_score', accuracy_score(y_test, rfc_pred), f1_score(y_test, rfc_pred, average = 'macro'), recall_score(y_test, rfc_pred, average = 'macro'))
	
	counts = y_train.value_counts().values
	X = pd.concat([X_train, y_train], axis=1)
	
	print('Reducing each class to 10% by maintaing the same ratios and reducing the overall size of dataset to its 10%')
	
	X_ = pd.concat([X[X['bin'] == 0].sample(int(0.1 * counts[0])), X[X['bin'] == 1].sample(int(0.1 * counts[1])), X[X['bin'] == 2].sample(int(0.1 * counts[2])), X[X['bin'] == 3].sample(int(0.1 * counts[3])), X[X['bin'] == 4].sample(int(0.1 * counts[4]))], axis=0)
	
	y_train = X_.bin
	X_train = X_.drop(['bin'], axis=1)
	
	rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
	rfc_pred = rfc.predict(X_test)
	print('Accuracy score f1_score recall_score', accuracy_score(y_test, rfc_pred), f1_score(y_test, rfc_pred, average = 'macro'), recall_score(y_test, rfc_pred, average = 'macro'))
	
	print('Reducing each class to the size of the smallest bin')
	X_ = pd.concat([X[X['bin'] == 0].sample(counts[4]), X[X['bin'] == 1].sample(counts[4]),X[X['bin'] == 2].sample(counts[4]),X[X['bin'] == 3].sample(counts[4]), X[X['bin'] == 4].sample(counts[4])], axis=0)
	y_train = X_.bin
	X_train = X_.drop(['bin'], axis=1)
	rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
	rfc_pred = rfc.predict(X_test)
	print('Accuracy score f1_score recall_score', accuracy_score(y_test, rfc_pred), f1_score(y_test, rfc_pred, average = 'macro'), recall_score(y_test, rfc_pred, average = 'macro'))
	
	print('Taking 1% the size of each class and oversampling to min(1% max bin size, bin_size)')
	X_ = pd.concat([X[X['bin'] == 0].sample(min(int(counts[0]/100), counts[0])), X[X['bin'] == 1].sample(min(int(counts[0]/100), counts[1])),X[X['bin'] == 2].sample(min(int(counts[0]/100), counts[2])),X[X['bin'] == 3].sample(min(int(counts[0]/100), counts[3])), X[X['bin'] == 4].sample(min(int(counts[0]/100), counts[4]))], axis=0)
	y_train = X_.bin
	X_train = X_.drop(['bin',], axis=1)
	
	rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
	rfc_pred = rfc.predict(X_test)
	print('Accuracy score f1_score recall_score', accuracy_score(y_test, rfc_pred), f1_score(y_test, rfc_pred, average = 'macro'), recall_score(y_test, rfc_pred, average = 'macro'))
	
	return
	
