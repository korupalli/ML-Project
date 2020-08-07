from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def DT(x_train,y_train,x_test,y_test):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return metrics.accuracy_score(y_test, y_pred)