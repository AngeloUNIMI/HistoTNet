import os
import matplotlib.pyplot as plt
from util.pause import pause


def dbToDataStore(dirIn, dirOut, extOrig, extNew, log):

    # display
    if log:
        print("Transforming DB...")

    # transform db
    for name in os.listdir(dirIn):
        if name.endswith(extOrig):
            # display
            #if log:
                #print("\tProcessing: " + name)
            # read name
            pre, ext = os.path.splitext(name)
            # get label
            C = pre.split('_')
            # create dir with label
            dirOutLabel = os.path.join(dirOut, C[1])
            # newname
            newName = pre + '.' + extNew
            newPath = os.path.join(dirOutLabel, newName)
            # if already present skip
            if os.path.exists(newPath):
                continue
            # create directory if not present
            if not os.path.exists(dirOutLabel):
                os.makedirs(dirOutLabel)
            # read
            img = plt.imread(os.path.join(dirIn, name))

            # display
            """
            print(newName)
            plt.imshow(img)
            plt.show()
            pause()
            """

            # write
            plt.imsave(newPath, img)

            #pause()

    print()

