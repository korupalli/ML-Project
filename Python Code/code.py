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
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

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
def select_k_best(n_features):
    f_score,k_best_columns,selected_col=f_values(train_set,n_features,'bin')
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

    cols = select_k_best(12)
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

    height2 = [f21.round(2), f22.round(2), f31.round(2), f32.round(2), f41.round(2), f42.round(2), f51.round(2),
              f52.round(2), f61.round(2), f62.round(2)]

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

#best models

def hypertuning():
    x_train=train_set.drop('bin',axis=1)
    y_train=train_set['bin']
    x_test=test_set.drop('bin',axis=1)
    y_test=test_set['bin']

    cols = select_k_best(2)
    X = train_set.drop('bin', axis=1)
    Y = test_set.drop('bin', axis=1)
    x_train_s = X.iloc[:, cols]
    x_test_s = Y.iloc[:, cols]


    a42, f42 = NB(x_train_s, y_train, x_test_s, y_test)
    print('Accuracy Score: NB classifier on selected features', a42, f42)

<<<<<<< HEAD
print(GaussianNB().get_params())
print(RandomForestClassifier().get_params())


# ['Favorites', 'Hashtags_count'] 2 naive bayes GaussianNB()
# ['Followers', 'Friends', 'Favorites', 'Month', 'Date', 'Neg', 'Mentions_count', 'Hashtags_count', 'Mentions_score', 'Hashtags_score', 'Mentions_score_avg', 'Hashtags_score_avg'] 12
# classification_compare()

# hypertuning()
=======
train_set=pd.read_csv('/content/drive/My Drive/Colab Notebooks/train.csv')
test_set=pd.read_csv('/content/drive/My Drive/Colab Notebooks/test.csv')

train_set.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1,inplace=True)
test_set.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1,inplace=True)
def hypertuning_Randomforest():
    rf = RandomForestClassifier()
    # Number of trees in random forest
    n_estimators = [5, 50, 250]
# Number of features to consider at every split
    max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(2, 50, num = 10)]
    max_depth.append(None)
# Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
    bootstrap = [True, False]
    criterion=['gini', 'entropy']
    max_leaf_nodes=[int(x) for x in np.linspace(10, 110, num = 30)]
    max_leaf_nodes.append(None)
    class_weight=['balanced', 'balanced_subsample']
    parameters = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion':criterion,
               'max_leaf_nodes':max_leaf_nodes,
               'class_weight':class_weight
               }
    Grid_cv = GridSearchCV(rf, parameters, cv=5)
    features=['Followers', 'Friends', 'Favorites', 'Month', 'Date', 'Neg', 'Mentions_count', 'Hashtags_count', 'Mentions_score', 'Hashtags_score', 'Mentions_score_avg', 'Hashtags_score_avg']
    Grid_cv.fit(train_set[features], train_set.bin)
    print(Grid_cv.best_params_)
    best_random = Grid_cv.best_estimator_
    random_accuracy = evaluate(best_random, test_set[features], test_set[bin])
    print( "Accuracy of Best Parameter",random_accuracy)
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
def hypertuning_GradientBoosting():
    rf = GradientBoostingClassifier()
    loss=["deviance", 'exponential']
    # Number of trees in random forest
    n_estimators = [5, 50, 250, 500]
# Number of features to consider at every split
    max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(2, 50, num = 10)]
    max_depth.append(None)
# Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
    bootstrap = [True, False]
    criterion=['friedman_mse', 'mse', 'mae']
    max_leaf_nodes=[int(x) for x in np.linspace(10, 110, num = 30)]
    max_leaf_nodes.append(None)
    class_weight=['balanced', 'balanced_subsample']
    parameters = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion':criterion,
               'max_leaf_nodes':max_leaf_nodes,
               'class_weight':class_weight
               }
    Grad_cv = GridSearchCV(rf, parameters, cv=5)
    features=['Followers', 'Friends', 'Favorites', 'Month', 'Date', 'Neg', 'Mentions_count', 'Hashtags_count', 'Mentions_score', 'Hashtags_score', 'Mentions_score_avg', 'Hashtags_score_avg']
    Grad_cv.fit(train_set[features], train_set.bin)
    print(Grad_cv.best_params_)
    best_random = Grad_cv.best_estimator_
    random_accuracy = evaluate(best_random, test_set[features], test_set[bin])
    print( "Accuracy of Best Parameter",random_accuracy)

# ['Favorites', 'Hashtags_count'] 2 naive bayes GaussianNB()
# ['Followers', 'Friends', 'Favorites', 'Month', 'Date', 'Neg', 'Mentions_count', 'Hashtags_count', 'Mentions_score', 'Hashtags_score', 'Mentions_score_avg', 'Hashtags_score_avg'] 12
#classification_compare()

#hypertuning()

hypertuning_Randomforest()
hypertuning_GradientBoosting()
>>>>>>> 85cd45ae16c254d5094b81217332f8b073d7e357
