import numpy as np
import shutil
import os


def proba_mass_split(y, folds):
    obs, classes = y.shape
    dist = y.sum(axis=0).astype('float')
    dist /= dist.sum()
    index_list = []
    fold_dist = np.zeros((folds, classes), dtype='float')
    for _ in range(folds):
        index_list.append([])
    for i in range(obs):
        if i < folds:
            target_fold = i
        else:
            normed_folds = fold_dist.T / fold_dist.sum(axis=1)
            how_off = normed_folds.T - dist
            target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(i)
    #print("Fold distributions are")
    #print(fold_dist)
    return index_list


def getIndexesCrossValMultiLab(classVec, foldsChosen):
    numSamples = len(classVec)
    numClasses = len(classVec[0])
    y = np.empty([numSamples, numClasses])
    for i, classV in enumerate(classVec):
        v_arr = np.array(classV, dtype=np.int32)
        y[i, :] = v_arr

    indexes = proba_mass_split(y, foldsChosen)

    return indexes


def splitDBTrainValTest(indexes, folderOrig, folderDest,
                        fileNameVec, dLabel, folds):

    types = ['train', 'val', 'test']

    # loop on types
    for type in types:
        # loop on folds
        for foldInd in folds[type]:
            indexesFold = indexes[foldInd]
            # loop on file indexes for fold
            for indexOne in indexesFold:
                shutil.copyfile(os.path.join(folderOrig, fileNameVec[indexOne]),
                                os.path.join(folderDest, type, dLabel, fileNameVec[indexOne]))


