# Apply PCA on the given BOW representation
print "Applying PCA on the data vector"
import numpy as np
from matplotlib.mlab import PCA
matrixData = np.array(vectors)
pcaVectors = PCA(matrixData)

# Cluster the top 20 dimensions of the PCA vectors
import nltk
from nltk import cluster
from nltk.cluster import cosine_distance
from nltk.cluster import KMeansClusterer

numClusters = 4
print "KMeans Clustering with %d means and using cosine distance" %numClusters
clusterVectors = [ x[:20] for x in pcaVectors.Y ]
clusterer = KMeansClusterer(numClusters, cosine_distance);
clusters = clusterer.cluster(clusterVectors, assign_clusters=True, trace=False);
means = clusterer.means();

# compute precision and recall
print "computing statistics for according to the given clusters"
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
