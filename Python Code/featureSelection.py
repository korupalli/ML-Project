from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import f_regression
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2

def chi2_test(frame,pred):


    X=frame.drop(pred,axis=1)
    y=frame[pred]

    chi_scores = chi2(X, y)
    chi_score=pd.Series(chi_scores[0], index=X.columns)
    chi_score.sort_values(ascending=False, inplace=True)
    p_values = pd.Series(chi_scores[1], index=X.columns)
    p_values.sort_values(ascending=False, inplace=True)

    return chi_score,p_values



def f_values(frame,value,pred):

    X = frame.drop(pred, axis=1)
    y = frame[pred]
    selector = SelectKBest(f_classif, k=value)
    selector.fit(X, y)
    cols = selector.get_support(indices=True)
    columns=X.columns
    selected_col=[columns[i] for i in cols]

    # fs = SelectKBest(score_func=f_regression, k=value)
    # apply feature selection
    # X_selected = fs.fit_transform(X, y)
    # print(X_selected)

    return selector.scores_,cols,selected_col

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

def correlate(x_train,method):
    corr_features = set()

    # create the correlation matrix (default to pearson)
    corr_matrix = x_train.corr(method=method)
    # print(corr_matrix)
    # optional: display a heatmap of the correlation matrix
    # plt.figure(figsize=(5, 5))
    # sns.heatmap(corr_matrix)
    # plt.title('Heat map: Spearman correlation coefficient')
    # plt.show()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:

                colname = corr_matrix.columns[i]
                corr_features.add(colname)

    return corr_features