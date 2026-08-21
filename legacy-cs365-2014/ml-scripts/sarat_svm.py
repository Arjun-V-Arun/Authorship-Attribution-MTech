##########################################################
# Analysis of the SVM that was performed on Sarat        #
#                                                        #
#                                                        #
# Requires: vectors                                      #
# Output  : count                                        #
# Usage   : %loadpy sarat_svm.py                         #
##########################################################

# Create the data
X = vectors[:500]                     # 500 correct
X.extend(vectors[702:1352])           # 650 incorrect
X.extend(vectors[1802:1952])          # 150 incorrect

# Create labels
y = []
for i in range(1300):
    if i < 500:
        y.append(1)
    else:
        y.append(0)

# Train SVM
from sklearn import svm
clf = svm.SVC()
clf.fit(X, y)
results = clf.predict(vectors)

# Count results
count = {'TP' : 0, 'FP': 0, 'TN': 0, 'FN':0 }
for i in range(2089):
    if i < 500: # 0, 499
        pass
    elif i < 702: # 500, 701
        if results[i]:
            count['TP'] += 1
        else:
            count['FN'] += 1
    elif i < 1352: # 702, 1351
        pass
    elif i < 1802: # 1352, 1801
        if results[i]:
            count['FP'] += 1
        else:
            count['TN'] += 1
        pass
    elif i < 1952: # 1802, 1951
        pass
    else: # 1952, 2088
        if results[i]:
            count['FP'] += 1
        else:
            count['TN'] += 1
        pass
