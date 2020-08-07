from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import f_regression
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
