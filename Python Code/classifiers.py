from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

def random_forest(x_train,y_train,x_test,y_test):
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return accuracy_score(y_test, y_pred),f1_score(y_test,y_pred,average='macro')

def DT(x_train,y_train,x_test,y_test):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return accuracy_score(y_test, y_pred),f1_score(y_test,y_pred,average='macro')



def KNN(x_train,y_train,x_test,y_test):
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return accuracy_score(y_test, y_pred),f1_score(y_test,y_pred,average='macro')

def NB(x_train,y_train,x_test,y_test):
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return accuracy_score(y_test, y_pred),f1_score(y_test,y_pred,average='macro')

def gradientBoost(x_train,y_train,x_test,y_test):
    clf = GradientBoostingClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return accuracy_score(y_test, y_pred),f1_score(y_test,y_pred,average='macro')

def MLP(x_train,y_train,x_test,y_test):
    clf = MLPClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return accuracy_score(y_test, y_pred),f1_score(y_test,y_pred,average='macro')

def LR(x_train,y_train,x_test,y_test):
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')

def ADB(x_train,y_train,x_test,y_test):
    clf = AdaBoostClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')