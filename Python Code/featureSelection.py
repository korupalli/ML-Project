from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

def chi2_test(frame,pred):

    columns=frame.columns
    for i in columns:
        frame[i]=LabelEncoder().fit_transform(frame[i])
    X=frame.drop(pred,axis=1)
    y=frame[pred]

    chi_scores = chi2(X, y)
    chi_score=pd.Series(chi_scores[0], index=X.columns)
    p_values = pd.Series(chi_scores[1], index=X.columns)

    bars = X.columns
    height = chi_score

    y_pos = np.arange(len(bars))
    # Create horizontal bars
    plt.barh(y_pos, height)
    # Create names on the y-axis
    plt.yticks(y_pos, bars)
    plt.title('chi-square')
    # Show graphic
    plt.show()

    bars = X.columns
    height = p_values
    y_pos = np.arange(len(bars))
    # Create horizontal bars
    plt.barh(y_pos, height)
    # Create names on the y-axis
    plt.yticks(y_pos, bars)
    plt.title('p-values')
    # Show graphic
    plt.show()

    return chi_score,p_values



def f_values(frame,value,pred):

    X = frame.drop(pred, axis=1)
    y = frame[pred]
    selector = SelectKBest(f_classif, k=value)
    selector.fit(X, y)
    cols = selector.get_support(indices=True)
    columns=X.columns
    selected_col=[columns[i] for i in cols]

    return selector.scores_,cols,selected_col

def correlate(x_train,method):
    corr_features = set()

    # create the correlation matrix (default to pearson)
    corr_matrix = x_train.corr(method=method)
    # print(corr_matrix)
    # optional: display a heatmap of the correlation matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(corr_matrix)
    plt.title('Heat map: Kendall correlation coefficient')
    plt.show()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.8:

                colname = corr_matrix.columns[i]
                corr_features.add(colname)

    return corr_features