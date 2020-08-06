from sklearn.feature_selection import SelectKBest, f_classif
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
    return selector.scores_,cols,selected_col
