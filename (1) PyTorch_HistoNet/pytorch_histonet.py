# --------------------------
# IMPORT
from torchvision import models
from torchvision import transforms
from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import warnings
# warnings.filterwarnings("ignore")
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
import split_folders
from random import seed
from random import random
from datetime import datetime
import pickle
import PIL

# --------------------------
# PRIV FUNCTIONS
import util
import functions
from modelGeno.vgg16_bn_Mahdi import vgg16_bn_Mahdi

# --------------------------
# CLASSES
from classes.classesADP import classesADP


# --------------------------
# PARAMS
from params.dirs import dirs


# --------------------------
# MAIN
if __name__ == '__main__':

    # params
    plotta = False
    log = True
    extOrig = 'png'
    extNew = 'png'
    num_iterations = 1
    nFolds = 10
    batch_sizeP = 32 #32
    batch_sizeP_norm = 1024
    numWorkersP = 0
    n_neighborsP = 1
    fontSize = 22
    padSize = 30
    num_epochs = 80 # 80


    # ------------------------------------------------------------------- db info
    dirWorkspaceOrig = './db_orig/'
    dirWorkspaceTest = './db_test/'
    dbName = 'ADP'
    ROI = 'img_res_1um_bicubic'
    csvFile = 'ADP_EncodedLabels_Release1_Flat.csv'


    # ------------------------------------------------------------------- Enable CUDA
    cuda = True if torch.cuda.is_available() else False
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Tensor = torch.cuda.DoubleTensor if cuda else torch.cuda.DoubleTensor
    if cuda:
        torch.cuda.empty_cache()
    print("Cuda is {0}".format(cuda))
    #util.pause()


    # ------------------------------------------------------------------- dirs
    dirsADP = dirs(dirWorkspaceOrig, dirWorkspaceTest, dbName, csvFile, ROI)
    dirDbOrig, dirDbTest, dirOutTrainTest = dirsADP.computeDirs()
    csvFileFull = dirsADP.getCSVfileFull()
    if not os.path.exists(dirDbTest):
        os.mkdir(dirDbTest)


    # ------------------------------------------------------------------- transform db
    dLabel = util.dbToDataStore(dirDbOrig, dirDbTest, extOrig, extNew, log)


    # ------------------------------------------------------------------- define all models we want to try
    modelNamesAll = list()
    modelNamesAll.append({'name': 'resnet18', 'sizeFeatures': 512})
    modelNamesAll.append({'name': 'vgg16_bn_Mahdi', 'sizeFeatures': 0})


    # ------------------------------------------------------------------- loop on hierarchies
    for hierarNum in range(0, 3):


        # ------------------------------------------------------------------- loop on models
        for i, (modelData) in enumerate(modelNamesAll):

            # dir results
            dirResult = './results/' + 'level_{}'.format(hierarNum+1) +  '/' + modelData['name'] + '/'
            if not os.path.exists(dirResult):
                os.makedirs(dirResult)

            # result file
            now = datetime.now()
            current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
            fileResultName = current_time + '.txt'
            fileResultNameFull = os.path.join(dirResult, fileResultName)
            fileResult = open(fileResultNameFull, "x")
            fileResult.close()

            # get all classes
            classVec, fileNameVec, columnNames = util.getAllClassesVec(hierarNum, classesADP, csvFileFull, log)

            # display
            if log:
                print()
                util.print_pers("Level: {0}".format(hierarNum+1), fileResultNameFull)
                util.print_pers("Model: {0}".format(modelData['name']), fileResultNameFull)

            # get current model
            if modelData['name'] == 'vgg16_bn_Mahdi':
                currentModel = vgg16_bn_Mahdi(classesADP[hierarNum]['numClasses'])
                imageSize = 224
            if modelData['name'] == 'resnet18':
                currentModel = models.resnet18(pretrained=False)
                new_fc = nn.Linear(modelData['sizeFeatures'], classesADP[hierarNum]['numClasses'])
                currentModel.fc = new_fc
                imageSize = 224
            #currentModel.double()
            # cuda
            if cuda:
                currentModel.to('cuda')
            # we want to train it
            for param in currentModel.parameters():
                param.requires_grad = True
            # log
            if log:
                print(currentModel)

            # optim
            optimizer_ft = optim.SGD(currentModel.parameters(), lr=0.02, momentum=0.9, weight_decay=0.0005)
            #optimizer_ft = optim.Adam(currentModel.parameters(), lr=0.02, weight_decay=0.0005)
            # cyclic LR
            exp_lr_scheduler = list()
            exp_lr_scheduler.append(lr_scheduler.CyclicLR(optimizer_ft, base_lr=0.001, max_lr=0.02,
                                                         mode='triangular', step_size_up=4, cycle_momentum=True))
            exp_lr_scheduler.append(lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.5))

            # preprocess
            transform = {
                'train':
                    transforms.Compose([
                    transforms.CenterCrop(272),
                    transforms.Resize(imageSize, interpolation=PIL.Image.BILINEAR),
                    #transforms.RandomRotation(45),
                    transforms.ToTensor()
                ]),
                'val':
                    transforms.Compose([  # [1]
                        transforms.CenterCrop(272),
                        transforms.Resize(imageSize, interpolation=PIL.Image.BILINEAR),
                        transforms.ToTensor()
                    ])
            }

            
            # cross val
            indexes = util.getIndexesCrossValMultiLab(classVec, nFolds)
            if os.path.exists(dirOutTrainTest):
                shutil.rmtree(dirOutTrainTest)
            # create
            if not os.path.exists(dirOutTrainTest):
                os.mkdir(dirOutTrainTest)
                os.mkdir(os.path.join(dirOutTrainTest, 'train'))
                os.mkdir(os.path.join(dirOutTrainTest, 'val'))
                os.mkdir(os.path.join(dirOutTrainTest, 'test'))
                os.mkdir(os.path.join(dirOutTrainTest, 'train', dLabel))
                os.mkdir(os.path.join(dirOutTrainTest, 'val', dLabel))
                os.mkdir(os.path.join(dirOutTrainTest, 'test', dLabel))
            # end cross val
            

            # ------------------------------------------------------------------- loop on iterations
            # init
            dataset_sizes = {}
            accuracyALL = np.zeros(num_iterations)
            for r in range(0, num_iterations): #(nFolds-1)

                # display
                if log:
                    util.print_pers("", fileResultNameFull)
                    util.print_pers("Iteration n. {0}".format(r + 1), fileResultNameFull)
                    util.print_pers("", fileResultNameFull)

                
                # define folds
                list_fold = list(range(0, nFolds))
                folds = {}
                fold_test = (nFolds-1)-r
                folds['test'] = [fold_test]
                # fold val
                if fold_test > 1:
                    fold_val = fold_test-1
                else:
                    fold_val = fold_test+1
                folds['val'] = [fold_val]
                # fold train - remaining
                list_fold_train = list_fold
                list_fold_train.remove(fold_test)
                list_fold_train.remove(fold_val)
                folds['train'] = list_fold_train
                # split db
                util.print_pers('Splitting DB', fileResultNameFull)
                util.splitDBTrainValTest(indexes, os.path.join(dirDbTest, dLabel), dirOutTrainTest,
                                         fileNameVec, dLabel, folds)
                # end define folds
                

                # get labels
                classVecPart = {}
                fileNameVecPart = {}
                classVecPart['train'], fileNameVecPart['train'] = \
                    util.extractLabels(os.path.join(dirOutTrainTest, 'train', dLabel), fileNameVec, classVec)
                classVecPart['val'], fileNameVecPart['val'] = \
                    util.extractLabels(os.path.join(dirOutTrainTest, 'val', dLabel), fileNameVec, classVec)


                # ------------------------------------------------------------------- TRAIN
                # load data
                # train
                all_idb2_train = datasets.ImageFolder(os.path.join(dirOutTrainTest, 'train'),
                                                           transform['train'])
                all_idb2_train_loader = torch.utils.data.DataLoader(all_idb2_train,
                                                                    batch_size=batch_sizeP_norm, shuffle=False,
                                                                    num_workers=numWorkersP, pin_memory=True)
                util.print_pers("\tClassi: {0}".format(all_idb2_train.classes), fileResultNameFull)
                dataset_sizes['train'] = len(all_idb2_train)
                util.print_pers("\tDimensione dataset train: {0}".format(dataset_sizes['train']), fileResultNameFull)

                # val
                all_idb2_val = datasets.ImageFolder(os.path.join(dirOutTrainTest, 'val'),
                                                         transform=transform['val'])
                all_idb2_val_loader = torch.utils.data.DataLoader(all_idb2_val,
                                                                  batch_size=batch_sizeP_norm, shuffle=False,
                                                                  num_workers=numWorkersP, pin_memory=True)
                util.print_pers("\tClassi: {0}".format(all_idb2_val.classes), fileResultNameFull)
                dataset_sizes['val'] = len(all_idb2_val)
                util.print_pers("\tDimensione dataset val: {0}".format(dataset_sizes['val']), fileResultNameFull)
                print()

                # mean, std
                print("Normalization...")
                # save norm
                fileNameSaveNorm = {}
                fileSaveNorm = {}
                meanNorm = {}
                stdNorm = {}
                dataloaders_all = list()
                dataloaders_all.append(all_idb2_train_loader)
                dataloaders_all.append(all_idb2_val_loader)
                dataset_sizes_all = dataset_sizes['train']+dataset_sizes['val']
                fileNameSaveNorm = os.path.join(dirResult, 'norm.dat')
                # if file exist, load
                if os.path.isfile(fileNameSaveNorm):
                    # read
                    fileSaveNorm = open(fileNameSaveNorm, 'rb')
                    meanNorm, stdNorm = pickle.load(fileSaveNorm)
                    fileSaveNorm.close()
                # else, compute normalization
                else:
                    # compute norm for all channels together
                    meanNorm, stdNorm = util.computeMeanStd(dataloaders_all, dataset_sizes_all, batch_sizeP_norm, cuda)
                    # save
                    fileSaveNorm = open(fileNameSaveNorm, 'wb')
                    pickle.dump([meanNorm, stdNorm], fileSaveNorm)
                    fileSaveNorm.close()

                # update transforms
                # train
                transform['train'] = transforms.Compose([
                    transforms.CenterCrop(272),
                    transforms.Resize(imageSize, interpolation=PIL.Image.BILINEAR),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    #transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[meanNorm, meanNorm, meanNorm],
                        std=[stdNorm, stdNorm, stdNorm]),
                ])
                # val
                transform['val'] = transforms.Compose([
                    transforms.CenterCrop(272),
                    transforms.Resize(imageSize, interpolation=PIL.Image.BILINEAR),
                    #transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[meanNorm, meanNorm, meanNorm],
                        std=[stdNorm, stdNorm, stdNorm],
                    )
                ])
                print()

                # update data loaders
                # train
                all_idb2_train = util.ImageFolderWithPaths(os.path.join(dirOutTrainTest, 'train'),
                                                           transformP=transform['train'],
                                                           classesP=classVecPart['train'],
                                                           filenamesP=fileNameVecPart['train'])
                all_idb2_train_loader = torch.utils.data.DataLoader(all_idb2_train,
                                                                    batch_size=batch_sizeP, shuffle=True,
                                                                    num_workers=numWorkersP, pin_memory=True)
                # val
                all_idb2_val = util.ImageFolderWithPaths(os.path.join(dirOutTrainTest, 'val'),
                                                         transformP=transform['val'],
                                                         classesP=classVecPart['val'],
                                                         filenamesP=fileNameVecPart['val'])
                all_idb2_val_loader = torch.utils.data.DataLoader(all_idb2_val,
                                                                    batch_size=batch_sizeP, shuffle=False,
                                                                    num_workers=numWorkersP, pin_memory=True)

                # train
                util.print_pers("Training", fileResultNameFull)
                # train net
                currentModel = functions.train_model_val(currentModel, classVecPart,
                                                         optimizer_ft, exp_lr_scheduler,
                                                         num_epochs, dataset_sizes, all_idb2_train_loader, all_idb2_val_loader,
                                                         batch_sizeP, classesADP[hierarNum]['numClasses'], modelData['name'],
                                                         dirResult, r, fileResultNameFull, log, cuda)


                # visualize some outputs
                #functions.visualize_model(currentModel, all_idb2_val_loader, cuda, columnNames, num_images=6)
                #util.pause()


                # ------------------------------------------------------------------- TEST
                torch.cuda.empty_cache()

                # display
                if log:
                    util.print_pers("Testing", fileResultNameFull)

                # eval
                currentModel.eval()
                # zero the parameter gradients
                optimizer_ft.zero_grad()
                torch.no_grad()

                # test transform
                transform['test'] = transforms.Compose([
                    transforms.CenterCrop(272),
                    transforms.Resize(imageSize, interpolation=PIL.Image.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[meanNorm, meanNorm, meanNorm],
                        std=[stdNorm, stdNorm, stdNorm],
                    )
                ])
                # load data
                all_idb2_test = datasets.ImageFolder(os.path.join(dirOutTrainTest, 'test'),
                                                     transform=transform['test'])
                all_idb2_test_loader = torch.utils.data.DataLoader(all_idb2_test,
                                                                   batch_size=batch_sizeP, shuffle=False,
                                                                   num_workers=numWorkersP)
                dataset_sizes['test'] = len(all_idb2_test)
                util.print_pers("\tDimensione dataset test: {0}".format(dataset_sizes['test']), fileResultNameFull)
                numBatches = {}
                numBatches['test'] = np.round(dataset_sizes['test'] / batch_sizeP)

                # labels
                classVecPart['test'], fileNameVecPart['test'] = \
                    util.extractLabels(os.path.join(dirOutTrainTest, 'test', dLabel), fileNameVec, classVec)

                # loop on images
                # init
                predALL_test = torch.zeros(dataset_sizes['test'], classesADP[hierarNum]['numClasses'])
                labelsALL_test = torch.zeros(dataset_sizes['test'], classesADP[hierarNum]['numClasses'])
                for batch_num, (inputs, y) in enumerate(all_idb2_test_loader):

                    ##################
                    #if batch_num > 10:
                        #break
                    ##################

                    # get size of current batch
                    sizeCurrentBatch = y.size(0)
                    #init tensor for labels
                    labelsTens = torch.zeros(sizeCurrentBatch, classesADP[hierarNum]['numClasses'], dtype=torch.float32)

                    if batch_num % 100 == 0:
                        print("\t\tBatch n. {0} / {1}".format(batch_num, int(numBatches['test'])))

                    if plotta:
                        util.visImage(inputs)
                        util.print_pers("\tClasse: {0}".format(y), fileResultNameFull)
                        # util.pause()

                    # stack
                    indStart = batch_num * batch_sizeP
                    indEnd = indStart + sizeCurrentBatch

                    # get labels from csv output
                    labelsVec = classVecPart['test'][indStart:indEnd]
                    for i, (listV) in enumerate(labelsVec):
                        # list to array
                        arr = np.array(listV, dtype=np.float32)
                        # array to tensor
                        tens = torch.tensor(arr)
                        # assign
                        labelsTens[i,:] = tens

                    # extract features
                    if cuda:
                        inputs = inputs.to('cuda')
                        labelsTens = labelsTens.to('cuda')

                    # predict
                    with torch.set_grad_enabled(False):
                        outputs = currentModel(inputs)
                        if cuda:
                            outputs = outputs.to('cuda')

                        m = nn.Sigmoid()
                        #preds = torch.round(m(outputs)).int()
                        preds = (m(outputs) > 0.5).int()

                        predALL_test[indStart:indEnd, :] = preds
                        labelsALL_test[indStart:indEnd, :] = labelsTens

                # end for x,y

                # accuracy
                accuracyResult = torch.sum(predALL_test == labelsALL_test).double()
                accuracyResult = accuracyResult / (dataset_sizes['test'] * classesADP[hierarNum]['numClasses'])

                # print(output_test)
                util.print_pers("\tAccuracy: {0:.2f}%".format(accuracyResult * 100), fileResultNameFull)

                # assign
                accuracyALL[r] = accuracyResult

                # newline
                util.print_pers("", fileResultNameFull)

                # save iter
                fileSaveIter = open(os.path.join(dirResult, 'results_{0}.dat'.format(r+1)), 'wb')
                pickle.dump([accuracyResult], fileSaveIter)
                fileSaveIter.close()
                fileSaveModelIter = open(os.path.join(dirResult, 'model_{0}.dat'.format(r+1)), 'wb')
                pickle.dump([currentModel], fileSaveModelIter)
                fileSaveModelIter.close()

            # end loop on iterations

            # average accuracy
            meanAccuracy = np.mean(accuracyALL)

            # display
            util.print_pers("Mean classification accuracy over {0} iterations; {1:.2f}".format(num_iterations, meanAccuracy), fileResultNameFull)
            util.print_pers("", fileResultNameFull)

            #close
            fileResult.close()

            # save
            fileSaveFinal = open(os.path.join(dirResult, 'resultsFinal.dat'), 'wb')
            pickle.dump([meanAccuracy], fileSaveFinal)
            fileSaveFinal.close()

            # del
            # torch.no_grad()
            del currentModel
            del all_idb2_train, all_idb2_train_loader
            del all_idb2_val, all_idb2_val_loader
            del all_idb2_test, all_idb2_test_loader
            del inputs, y
            torch.cuda.empty_cache()




