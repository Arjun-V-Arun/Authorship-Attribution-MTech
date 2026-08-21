######################################
# Cluster a BOW vector in 4 clusters #
#                                    #
# Requirements: clusterVectors       #
# Usage       : %loadpy cluster.py   #
######################################

import nltk
from nltk import cluster
from nltk.cluster import cosine_distance
from nltk.cluster import KMeansClusterer

numClusters = 4
print "KMeans Clustering with %d means and using cosine distance" %numClusters
clusterer = KMeansClusterer(numClusters, cosine_distance);
clusters = clusterer.cluster(clusterVectors, assign_clusters=True, trace=False);
means = clusterer.means();
