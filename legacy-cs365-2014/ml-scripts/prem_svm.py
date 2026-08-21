##########################################################
# Analysis of the SVM that was performed on Prem         #
#                                                        #
#                                                        #
# Requires: vectors                                      #
# Output  : count                                        #
# Usage   : %loadpy prem_svm.py                          #
##########################################################

# Create the data
X = vectors[:500] # 500 false
X.extend(vectors[1181:1581]) # 400 true
X.extend(vectors[1852:2052]) # 200 false

# Label the data
y = []
for i in range(1100):
    if i < 500:
        y.append(0)
    elif i < 900:
        y.append(1)
    else:
        y.append(0)

# Train SVM
from sklearn import svm
clf = svm.SVC()
clf.fit(X, y)
results = clf.predict(vectors)

# Count the results
count = {'TP' : 0, 'FP': 0, 'TN': 0, 'FN':0 }
for i in range(2089):
    if i < 500: # 0, 499
        pass
    elif i < 1181: # 500, 1180
        if results[i]:
            count['FP'] += 1
        else:
            count['TN'] += 1
    elif i < 1581: # 1181, 1580
        pass
    elif i < 1839: # 1581, 1838
        if results[i]:
            count['TP'] += 1
        else:
            count['FN'] += 1
    elif i < 1852: # 1849, 1851 
        if results[i]:
            count['FP'] += 1
        else:
            count['TN'] += 1
    elif i < 2052: # 1852, 2051
        pass
    else:
        if results[i]:
            count['FP'] += 1
        else:
            count['TN'] += 1
