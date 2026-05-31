import numpy as np

def computeCMC(distMatrix, TestLabels, padSize):

    sizeDistances = distMatrix.shape
    rankV = np.empty((sizeDistances[0]), np.float64)

    for r in range(0, sizeDistances[0]):
        distV = distMatrix[r, :].tolist()
        sortV = np.sort(distV).tolist()

        del sortV[0]
        minD = sortV[0]
        idx = distV.index(minD)

        rankV[r] = 1

        while (TestLabels[idx] != TestLabels[r]) and (len(sortV) >= 2):
            del sortV[0]
            minD = sortV[0]
            idx = distV.index(minD)

            rankV[r] = rankV[r] + 1

    # cast
    rankV = rankV.astype(int).tolist()

    listA = list(range(1, int(np.max(rankV))+1))
    probRanks = np.empty((len(listA)), np.float64)
    for ll in range(0, len(listA)):
        probRanks[ll] = rankV.count(listA[ll]) / sizeDistances[0]

    #print(probRanks)

    cmcV = np.empty((len(listA)), np.float64)
    for i in range(0, len(probRanks)):
        if i == 0:
            cmcV[i] = probRanks[i]
        else:
            cmcV[i] = cmcV[i-1] + probRanks[i]

    # pad
    cmcV_pad = np.pad(cmcV, (0, padSize-len(cmcV)), 'edge')

    #print(cmcV)

    #print(np.max(rankV))
    #print(listA)

    return cmcV_pad


