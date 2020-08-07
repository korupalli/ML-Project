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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score
from sklearn.ensemble import RandomForestClassifier
from chi_squared import chi2_test
from f_values import f_values
from vif import variance_inflation_factor
from dt_classifier import DT
import sys

train_set=pd.read_csv('train.csv')
test_set=pd.read_csv('test.csv')

train_set.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1,inplace=True)
test_set.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1,inplace=True)

print(train_set.columns)
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
    f_score,k_best_columns,selected_col=f_values(train_set,5,'bin')
    print(selected_col)
    return k_best_columns

def decision_tree():
    x_train=train_set.drop('bin',axis=1)
    y_train=train_set['bin']
    x_test=test_set.drop('bin',axis=1)
    y_test=test_set['bin']

    accuracy=DT(x_train,y_train,x_test,y_test)
    print('Accuracy Score: Decision Tree classifier on all features',accuracy)

    cols=select_k_best()
    X=train_set.drop('bin',axis=1)
    Y=test_set.drop('bin',axis=1)
    x_train_s = X.iloc[:,cols]
    x_test_s = Y.iloc[:,cols]
    bestk_accuracy=DT(x_train_s,y_train,x_test_s,y_test)
    print('Accuracy Score: Decision Tree classifier on selected features',bestk_accuracy)




# chi_square()
decision_tree()
