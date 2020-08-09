import pandas as pd, numpy as np, os, gc, matplotlib.pyplot as plt, seaborn as sb, re, warnings, calendar, sys
from copy import deepcopy
import statsmodels.api as sm
from math import log2
from collections import defaultdict
import seaborn as sns; sns.set()
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

from featureSelection import f_values,chi2_test,correlate
from vif import variance_inflation_factor

from classifiers import KNN,random_forest,DT,NB,gradientBoost,MLP,LR,ADB

import sys

train_set=pd.read_csv('train.csv')
test_set=pd.read_csv('test.csv')

train_set.drop(['Tweet Id'],axis=1,inplace=True)
test_set.drop(['Tweet Id'],axis=1,inplace=True)

# print(len(train_set.columns))
# print(test_set.columns)


# Feature Selection
# chi-square scores and p-values.
def chi_square(train_set):
    x_train=train_set
    chi_scores,p_values=chi2_test(x_train,'bin')

#SelectKbest features based on f-values.
def select_k_best(n_features):
    f_score,k_best_columns,selected_col=f_values(train_set,n_features,'bin')
    # print(selected_col,len(selected_col))
    return k_best_columns

def correlation(train_set,method='kendall'):
    x_train=train_set.drop('bin',axis=1)
    corr_features=correlate(x_train,method)
    # print(corr_features)

def classification_compare(train_set,test_set):

    x_train=train_set.drop('bin',axis=1)
    y_train=train_set['bin']
    x_test=test_set.drop('bin',axis=1)
    y_test=test_set['bin']

    cols = select_k_best(12)
    X = train_set.drop('bin', axis=1)
    Y = test_set.drop('bin', axis=1)
    x_train_s = X.iloc[:, cols]
    x_test_s = Y.iloc[:, cols]

    a21,f21=random_forest(x_train,y_train,x_test,y_test)
    # print('Accuracy Score: RandomForest classifier on all features', a21,f21)

    a22,f22 = random_forest(x_train_s,y_train,x_test_s,y_test)
    # print('Accuracy Score: RandomForest classifier on selected features', a22,f22)

    # best features=2
    a31, f31 = KNN(x_train, y_train, x_test, y_test)
    # print('Accuracy Score: KNN classifier on all features', a31, f31)

    a32, f32 = KNN(x_train_s, y_train, x_test_s, y_test)
    # print('Accuracy Score: KNN classifier on selected features', a32, f32)

    # best =2 or use 5
    a41, f41 = NB(x_train, y_train, x_test, y_test)
    # print('Accuracy Score: NB classifier on all features', a41, f41)

    a42, f42 = NB(x_train_s, y_train, x_test_s, y_test)
    # print('Accuracy Score: NB classifier on selected features', a42, f42)

    a51, f51 = gradientBoost(x_train, y_train, x_test, y_test)
    # print('Accuracy Score: gradient boost classifier on all features', a51, f51)

    a52, f52 = gradientBoost(x_train_s, y_train, x_test_s, y_test)
    # print('Accuracy Score: gradient boost classifier on selected features', a52, f52)

    # k=12
    a61, f61 = MLP(x_train, y_train, x_test, y_test)
    # print('Accuracy Score: MLP on all features', a61, f61)

    a62, f62 = MLP(x_train_s, y_train, x_test_s, y_test)
    # print('Accuracy Score: MLP on selected features', a62, f62)

    a71, f71 = ADB(x_train, y_train, x_test, y_test)
    # print('Accuracy Score: MLP on all features', a61, f61)

    a72, f72 = ADB(x_train_s, y_train, x_test_s, y_test)
    # print('Accuracy Score: MLP on selected features', a62, f62)

    bars = ('RF', 'RF_k', 'KNN', 'KNN_k', 'NaiveBayes', 'NaiveBayes_k', 'GB', 'GB_k', 'MLP', 'MLP_k','ADB','ADB_k')
    height = [a21.round(2),a22.round(2),a31.round(2),a32.round(2),a41.round(2),a42.round(2),a51.round(2),a52.round(2),a61.round(2),a62.round(2), a71.round(2), a72.round(2)]

    height2 = [f21.round(2), f22.round(2), f31.round(2), f32.round(2), f41.round(2), f42.round(2), f51.round(2),
              f52.round(2), f61.round(2), f62.round(2), f71.round(2), f72.round(2)]

    y_pos = np.arange(len(bars))

    # Create horizontal bars
    plt.barh(y_pos, height)

    # Create names on the y-axis
    plt.yticks(y_pos, bars)
    plt.title('Classifiers Accuracy Comparison: All features vs K best features')
    # Show graphic
    plt.show()

    y_pos = np.arange(len(bars))

    # Create horizontal bars
    plt.barh(y_pos, height2)

    # Create names on the y-axis
    plt.yticks(y_pos, bars)
    plt.title('Classifiers f1-score Comparison: All features vs K best features')
    # Show graphic
    plt.show()

# chi_square(train_set)
# correlation(train_set)
classification_compare(train_set,test_set)
