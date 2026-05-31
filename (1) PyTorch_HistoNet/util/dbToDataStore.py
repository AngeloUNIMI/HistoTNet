import os
import matplotlib.pyplot as plt
from util.pause import pause
import csv
import torch
from torchvision import datasets


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, image_pathP, transformP, classesP, filenamesP):
        super(ImageFolderWithPaths, self).__init__(root=image_pathP, transform=transformP)
        self.data = datasets.ImageFolder(image_pathP, transformP)
        self.classVec = classesP
        self.fileNameVec = filenamesP

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        im, dumTarget = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]


        #labelsTens = list()
        #for i, (s_path) in enumerate(path):
            # label
        classV = ()
        dir, filename = os.path.split(path)
        indexL = self.fileNameVec.index(filename)
        classV = self.classVec[indexL]
        #labelsTens.append(classV)

        #print(index)
        #print(path)
        #print(torch.tensor(classV))
        #pause()

        # make a new tuple that includes original and the path
        #tuple_with_path = (original_tuple + (path,))
        #return tuple_with_path

        return im, dumTarget, (path), (torch.tensor(classV))


def getClass(columnNames, rowF, hierarNum, classesADP):
    classes = list()
    for className in classesADP[hierarNum]['classesNames']:
        classes.append(rowF[columnNames.index(className)])
    #print(classes)
    #classOne = torch.max(classes, 1)
    # cast
    classesInt = [int(i) for i in classes]
    return classesInt


def getAllClassesVec(hierarNum, classesADP, csvFileFull, log):
    # open csv
    allClasses = list()
    allFileNames = list()
    with open(csvFileFull) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                columnNames = row
            else:
                #print(row)
                fileName = row[0]
                classVec = getClass(columnNames, row, hierarNum, classesADP)
                allClasses.append(classVec)
                allFileNames.append(fileName)
                #print(classVec)
                #pause()
            line_count += 1
        print('Processed {line_count} lines.')
    return allClasses, allFileNames, columnNames

def dbToDataStore(dirIn, dirOut, extOrig, extNew, log):

    dLabel = 'dummyLabel'

    # display
    if log:
        print("Transforming DB...")

    # dummy dir
    dirOutLabel = os.path.join(dirOut, dLabel)
    # create directory if not present
    if not os.path.exists(dirOutLabel):
        os.mkdir(dirOutLabel)

    # transform db
    for i, name in enumerate(os.listdir(dirIn)):
        if name.endswith(extOrig):
            # display
            if log and (i % 1000 == 0):
                print("\tProcessing: " + name)
            # read name
            pre, ext = os.path.splitext(name)
            # newname
            newName = pre + '.' + extNew
            newPath = os.path.join(dirOutLabel, newName)
            # if already present skip
            if os.path.exists(newPath):
                continue
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

    return dLabel


def extractLabels(dirDbPart, fileNameVec, classVec):
    classVall = list()
    fileNameVecAll = list()
    for i, name in enumerate(os.listdir(dirDbPart)):
        indexL = fileNameVec.index(name)
        fileNameVecAll.append(name)
        classV = classVec[indexL]
        classVall.append(classV)
    return classVall, fileNameVecAll