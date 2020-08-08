import pandas as pd
import numpy as np
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd, numpy as np, os, gc, matplotlib.pyplot as plt, seaborn as sb, re, warnings, calendar, sys
from copy import deepcopy
import statsmodels.api as sm
from math import log2
from collections import defaultdict
import seaborn as sns; sns.set()
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score
from sklearn.ensemble import RandomForestClassifier

from featureSelection import f_values,chi2_test,correlate
from vif import variance_inflation_factor

from classifiers import KNN,random_forest,DT,NB,gradientBoost,MLP,LR

import sys

train_set=pd.read_csv('train.csv')
test_set=pd.read_csv('test.csv')

train_set.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1,inplace=True)
test_set.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1,inplace=True)

print(len(train_set.columns))
print(test_set.columns)


# Feature Selection
# chi-square scores and p-values.
def chi_square():
    x_train=train_set.drop('Neg',axis=1)
    chi_scores,p_values=chi2_test(x_train,'bin')
    chi_scores.plot.bar()
    plt.title('Chi-Square')
    plt.show()
    p_values.plot.bar()
    plt.title('p-values')
    plt.show()

#SelectKbest features based on f-values.
def select_k_best():
    f_score,k_best_columns,selected_col=f_values(train_set,12,'bin')
    print(selected_col,len(selected_col))
    return k_best_columns

def correlation(method):
    x_train=train_set.drop('bin',axis=1)
    corr_features=correlate(x_train,method)
    print(corr_features)

def classification_compare():
    x_train=train_set.drop('bin',axis=1)
    y_train=train_set['bin']
    x_test=test_set.drop('bin',axis=1)
    y_test=test_set['bin']

    cols = select_k_best()
    X = train_set.drop('bin', axis=1)
    Y = test_set.drop('bin', axis=1)
    x_train_s = X.iloc[:, cols]
    x_test_s = Y.iloc[:, cols]

    #DT k best = 5
    # a11,f11=DT(x_train,y_train,x_test,y_test)
    # print('Accuracy Score: Decision Tree classifier on all features',a11,f11)
    #
    # a12,f12=DT(x_train_s,y_train,x_test_s,y_test)
    # print('Accuracy Score: Decision Tree classifier on selected features',a12,f12)
    classifier=['RF','RF_k','KNN','KNN_k','NaiveBayes','NaiveBayes_k','GB','GB_k','MLP','MLP_k']
    accuracy=[]
    #random forest k best=13
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

    bars = ('RF', 'RF_k', 'KNN', 'KNN_k', 'NaiveBayes', 'NaiveBayes_k', 'GB', 'GB_k', 'MLP', 'MLP_k')
    height = [a21.round(2),a22.round(2),a31.round(2),a32.round(2),a41.round(2),a42.round(2),a51.round(2),a52.round(2),a61.round(2),a62.round(2)]


    y_pos = np.arange(len(bars))

    # Create horizontal bars
    plt.barh(y_pos, height)

    # Create names on the y-axis
    plt.yticks(y_pos, bars)
    plt.title('Classifiers Accuracy Comparison: All features vs K best features')
    # Show graphic
    plt.show()




classification_compare()