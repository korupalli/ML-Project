import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,classification_report

def ensembler(gb, nb, rfc):
	for i in range(len(gb)):
		if gb[i]== nb[i] or gb[i] == rfc[i]:
			nb[i] = gb[i]
		elif nb[i] == rfc[i]:
			continue
		else:
			nb[i] = gb[i]
	return nb


def Hyper_Tuning(train, test):
	features=['Followers', 'Friends', 'Favorites', 'Month', 'Date', 'Neg', 'Mentions_count', 'Hashtags_count', 'Mentions_score', 'Hashtags_score', 'Mentions_score_avg', 'Hashtags_score_avg']

	y_train = train.bin
	X_train = train[features]
	y_test = test.bin
	X_test = test[features]
	#For gradient boosting

	param_test2 = {'learning_rate':[0.1,0.15]#,'n_estimators':range(20,100,20),
               #'max_features':range(2,5),
               #'max_depth':range(4,12,2), 'min_samples_split':range(40,70,10),'min_samples_leaf':range(10,20,2)
              }
	gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_features=3,min_samples_split=15,min_samples_leaf=7,max_depth=4,subsample=0.8,random_state=10),
	param_grid = param_test2, scoring='accuracy',n_jobs=4, cv=3)
	gsearch2.fit(X_train[features],y_train)

	print('Gradient Boosting')
	print('Grid Search Result - ')

	print(gsearch2.best_params_, gsearch2.best_score_, '\n')

	print('Test set result')
	GB = GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_features=3,min_samples_split=50,min_samples_leaf=7,max_depth=5,subsample=0.8,random_state=10).fit(X_train, y_train)
	GB_pred = GB.predict(X_test)
	print('Accuracy    F1_score    Recall')
	print(accuracy_score(y_test, GB_pred), f1_score(y_test, GB_pred, average = 'macro'), recall_score(y_test, GB_pred, average = 'macro'),'\n')
	
	#Naive Bayes
	clf = GaussianNB()
	clf.fit(X_train[['Favorites', 'Hashtags_count']], y_train)
	y_pred = clf.predict(X_test[['Favorites', 'Hashtags_count']])
	print('Naive Bayes')
	print('Test set result')
	print('Accuracy    F1_score    Recall')
	print(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average = 'macro'), recall_score(y_test, y_pred, average = 'macro'),'\n')

	#Randon Forest classifier
	parameters = {'max_depth':range(7,15,2)}#,'n_estimators':range(200,220,5),'max_features':[4], 'criterion':['gini','entropy'],'class_weight':['balanced'],'min_samples_split':range(2,5)}
	Grid_cv = GridSearchCV(RandomForestClassifier(n_estimators=215,max_features=4,max_depth=13,criterion='entropy',min_samples_split=3,n_jobs=2), parameters,scoring='accuracy', cv=3)
	Grid_cv.fit(X_train[features],y_train)
	
	print('Random Forest')
	print('Random Forest result - ')
	print(Grid_cv.best_params_, Grid_cv.best_score_,'\n')

	rfc = RandomForestClassifier(n_estimators=215,max_features=4,max_depth=13,criterion='entropy',min_samples_split=3,n_jobs=2).fit(X_train, y_train)
	rfc_pred = rfc.predict(X_test)
	print('Accuracy    F1_score    Recall')
	print(accuracy_score(y_test, rfc_pred), f1_score(y_test, rfc_pred, average = 'macro'), recall_score(y_test, rfc_pred, average = 'macro'),'\n')
	
	ensemble_pred = ensembler(GB_pred, y_pred, rfc_pred)
	print('Ensemble Model: Random Forest, Naive Bayes, Gradient Boost')
	print('Accuracy    			F1_score    		Recall')
	print(accuracy_score(y_test, ensemble_pred), f1_score(y_test, ensemble_pred, average = 'macro'), recall_score(y_test, ensemble_pred, average = 'macro'))
	target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
	print(classification_report(y_test, ensemble_pred, target_names=target_names))
	
	return None
	
