
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

LR = LogisticRegression(random_state = 0)
SV = SVC(kernel = 'rbf', random_state = 0)
DT = DecisionTreeClassifier()
RF = RandomForestClassifier(n_estimators=105,criterion='entropy', min_samples_split=25, max_depth=7, max_features=None)

#-----------------------------------
LR.fit(X_train,Y_train)

y_pred = LR.predict(X_test)
  
accuracy = metrics.accuracy_score(y_pred,Y_test)
print ("LR_Accuracy : %s" % "{0:.2%}".format(accuracy))

lrcm = confusion_matrix(y_pred, Y_test)

#--------------------------

SV.fit(X_train,Y_train)
y_pred = SV.predict(X_test)
  
accuracy = metrics.accuracy_score(y_pred,Y_test)
print ("SV_Accuracy : %s" % "{0:.2%}".format(accuracy))

svcm = confusion_matrix(y_pred, Y_test)

#--------------------------

DT.fit(X_train,Y_train)
y_pred = DT.predict(X_test)
  
accuracy = metrics.accuracy_score(y_pred,Y_test)
print ("DT_Accuracy : %s" % "{0:.2%}".format(accuracy))

dtcm = confusion_matrix(y_pred, Y_test)

#-----------------------------

RF.fit(X_train,Y_train)
y_pred = RF.predict(X_test)

accuracy = metrics.accuracy_score(y_pred,Y_test)
print ("RF_Accuracy : %s" % "{0:.2%}".format(accuracy))

rfcm = confusion_matrix(y_pred, Y_test)

featimp = pd.Series(RF.feature_importances_, index=fe).sort_values(ascending=False)
print (featimp)

#-------------------------------

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = RF, X = X_train, y = Y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())