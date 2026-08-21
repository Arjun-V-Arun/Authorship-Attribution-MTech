# This file counts the various counts for different authors, but needs to have a 
# clusters vector which stores the cluster corresponding to the vectors array

# Note, here the arrangement of authors is according to the way glob.glob globs the 
# files while processing them to convert them into BOW
# sarat > vibhuti > prem > dharamvir

# This is not a very generic file and only works for the given dataset

# Needs: clusters


# compute precision and recall
measurements = {
        "sarat": { "cluster": 0, "recall": 0, "precision": 0, "number": 702, "count" : [0 for x in range(numClusters)]},
        "vibhuti": {"cluster": 0, "recall": 0, "precision": 0, "number": 479, "count" : [0 for x in range(numClusters)]},
        "prem": {"cluster": 0, "recall": 0, "precision": 0, "number": 658, "count" : [0 for x in range(numClusters)]},
        "dharamvir" : {"cluster": 0, "recall": 0, "precision": 0, "number": 250, "count" : [0 for x in range(numClusters)]}
        }

totalSnippets = 0
for key in measurements.keys():
    totalSnippets = totalSnippets + measurements[key]["number"]

for i in range(totalSnippets):
    if i < measurements["sarat"]["number"]:
        for j in range(numClusters):
            if clusters[i] == j:
                measurements["sarat"]["count"][j] = measurements["sarat"]["count"][j] + 1
                break
    elif i < measurements["sarat"]["number"] + measurements["vibhuti"]["number"]:
        for j in range(numClusters):
            if clusters[i] == j:
                measurements["vibhuti"]["count"][j] = measurements["vibhuti"]["count"][j] + 1
                break
    elif i < measurements["sarat"]["number"] + measurements["vibhuti"]["number"] + measurements["prem"]["number"]:
        for j in range(numClusters):
            if clusters[i] == j:
                measurements["prem"]["count"][j] = measurements["prem"]["count"][j] + 1
                break
    else:
        for j in range(numClusters):
            if clusters[i] == j:
                measurements["dharamvir"]["count"][j] = measurements["dharamvir"]["count"][j] + 1
                break

# computing the ratios
def computeRatio():
    for key in measurements.keys():
        measurements[key]["countRatio"] = [0 for x in range(numClusters)]
        for i in range(numClusters):
            measurements[key]["countRatio"][i] = measurements[key]["count"][i]/measurements[key]["number"]

# find the cluster for each author
def computeCluster():
    for key in measurements.keys():
        l = list(measurements[key]["count"])
        l.sort()
        measurements[key]["cluster"] = measurements[key]["count"].index(l.pop())

# compute precision and recall
computeCluster()
sumCluster = []
for i in range(numClusters):
    tempSum = 0
    for key in measurements.keys():
        tempSum = tempSum + measurements[key]["count"][i]
    sumCluster.append(tempSum)

for key in measurements.keys():
    cluster = measurements[key]["cluster"]
    measurements[key]["recall"] = measurements[key]["count"][cluster]/ float(measurements[key]["number"])
    measurements[key]["precision"] = measurements[key]["count"][cluster]/float(sumCluster[cluster])
