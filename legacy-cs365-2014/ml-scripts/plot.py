################################################
# Plot three PCA components                    #
#                                              #
# Requires: clusters, pcaVectors, measurements #
################################################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = [[] for i in range(numClusters + 1)]
y = [[] for i in range(numClusters + 1)]
z = [[] for i in range(numClusters + 1)]

plt.close('all') # close all latent plotting windows
fig1 = plt.figure() # Make a plotting figure

ax = Axes3D(fig1) # use the plotting figure to create a Axis3D object.

# Iterate over all the vectors
# We classify vectors according to the four authors plus misclassfication
i = -1
for item in pcaVectors.Y:
    i = i + 1
    if i < measurements["sarat"]["number"]:
        if clusters[i] == measurements['sarat']['cluster']:
            x[0].append(item[0])
            y[0].append(item[1])
            z[0].append(item[2])
        else:
            x[4].append(item[0])
            y[4].append(item[1])
            z[4].append(item[2])
    elif i < measurements["sarat"]["number"] + measurements["vibhuti"]["number"]:
        if clusters[i] == measurements['vibhuti']['cluster']:
            x[1].append(item[0])
            y[1].append(item[1])
            z[1].append(item[2])
        else:
            x[4].append(item[0])
            y[4].append(item[1])
            z[4].append(item[2])
    elif i < measurements["sarat"]["number"] + measurements["vibhuti"]["number"] + measurements["prem"]["number"]:
        if clusters[i] == measurements['prem']['cluster']:
            x[2].append(item[0])
            y[2].append(item[1])
            z[2].append(item[2])
        else:
            x[4].append(item[0])
            y[4].append(item[1])
            z[4].append(item[2])
    else:
        if clusters[i] == measurements['dharamvir']['cluster']:
            x[3].append(item[0])
            y[3].append(item[1])
            z[3].append(item[2])
        else:
            x[4].append(item[0])
            y[4].append(item[1])
            z[4].append(item[2])

pltData = [x[0],y[0],z[0]] 
ax.scatter(pltData[0], pltData[1], pltData[2], c='#4169e1') # make a scatter plot of blue dots from the data

pltData = [x[1],y[1],z[1]] 
ax.scatter(pltData[0], pltData[1], pltData[2], c='#00ff7f') # make a scatter plot of blue dots from the data

pltData = [x[2],y[2],z[2]] 
ax.scatter(pltData[0], pltData[1], pltData[2], c='#ff6347') # make a scatter plot of blue dots from the data

pltData = [x[3],y[3],z[3]] 
ax.scatter(pltData[0], pltData[1], pltData[2], c='y') # make a scatter plot of blue dots from the data

pltData = [x[4],y[4],z[4]] 
ax.scatter(pltData[0], pltData[1], pltData[2], c='#ff1493') # make a scatter plot of blue dots from the data
# ax.scatter(pltData[0], pltData[1], pltData[2], c='k') # make a scatter plot of blue dots from the data
 
# make simple, bare axis lines through space:
xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0), (0,0)) # 2 points make the x-axis line at the data extrema along x-axis 
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k') # make a red line for the x-axis.
yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])), (0,0)) # 2 points make the y-axis line at the data extrema along y-axis
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k') # make a red line for the y-axis.
zAxisLine = ((0, 0), (0,0), (min(pltData[2]), max(pltData[2]))) # 2 points make the z-axis line at the data extrema along z-axis
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k') # make a red line for the z-axis.
 
# label the axes 
ax.set_xlabel("First Principal Component") 
ax.set_ylabel("Second Principal Component")
ax.set_zlabel("Third Principal Component")
ax.set_title("Combined Ngram KMeans clustering")
plt.show() # show the plot
