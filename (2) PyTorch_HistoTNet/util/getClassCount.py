import numpy as np
from util import pause

def getClassCount(classVec):
    #print(classVec[0])
    classCounts = np.zeros(len(classVec[0]))
    for vec in classVec:
        classCounts += vec

        #print(vec)

    #print('Uscita dalla funzione classcount')
    #print(classCounts)
    #pause()

    return classCounts

