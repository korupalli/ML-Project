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


