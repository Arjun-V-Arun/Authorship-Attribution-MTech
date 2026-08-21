#################################################
# SVM routine                                   #
#                                               #
# Requirements: data-matrix(X), labels(y)       #
# Usage       : %loadpy svm.py                  #
#################################################

from sklearn import svm
clf = svm.SVC()
clf.fit(X, y)
results = clf.predict(vectors)
