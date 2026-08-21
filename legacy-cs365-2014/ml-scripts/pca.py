######################################
# PCA on any given feature matrix    #
#                                    #
# Requirements: vectors array        #
# Usage       : %loadpy pca.py       #
######################################

import numpy as np
from matplotlib.mlab import PCA
matrixData = np.array(vectors)
pcaVectors = PCA(matrixData)
