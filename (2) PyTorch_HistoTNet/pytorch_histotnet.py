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
# MAIN
if __name__ == '__main__':

    # params
    plotta = False
    log = True
    extOrig = 'tif'
    extNew = 'png'
    num_iterations = 10
    # nFolds = 10
    batch_sizeP = 8 #32
    batch_sizeP_norm = 8
    numWorkersP = 0
    n_neighborsP = 1
    fontSize = 22
    padSize = 30
    num_epochs = 100


    # ------------------------------------------------------------------- db info
    dirWorkspace = './db/'
    dirPretrainedModels = './pretrained_nets/'
    dbName = 'ALL_IDB2'
    ROI = 'ROI'


    # ------------------------------------------------------------------- Enable CUDA
    cuda = True if torch.cuda.is_available() else False
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Tensor = torch.cuda.DoubleTensor if cuda else torch.cuda.DoubleTensor
    if cuda:
        torch.cuda.empty_cache()
    print("Cuda is {0}".format(cuda))
    #util.pause()


    # ------------------------------------------------------------------- dirs
    dirDbOrig = dirWorkspace + dbName + '/'
    dirDbTest = dirWorkspace + dbName + '/datastore/'
    dirOutTrainTest = dirWorkspace + dbName + '/datastore_trainTest/'
    if not os.path.exists(dirDbTest):
        os.makedirs(dirDbTest)
    if not os.path.exists(dirOutTrainTest):
        os.makedirs(dirOutTrainTest)


    # ------------------------------------------------------------------- transform db
    util.dbToDataStore(dirDbOrig, dirDbTest, extOrig, extNew, log)


    # ------------------------------------------------------------------- define all models we want to try
    modelNamesAll = list()
    modelNamesAll.append({'name': 'resnet18', 'sizeFeatures': 512})
    modelNamesAll.append({'name': 'vgg16_bn_Mahdi', 'sizeFeatures': 512})


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

            # display
            if log:
                print()
                util.print_pers("Level: {0}".format(hierarNum+1), fileResultNameFull)
                util.print_pers("Model: {0}".format(modelData['name']), fileResultNameFull)


            # ------------------------------------------------------------------- loop on iterations
            # init
            dataset_sizes = {}
            accuracyALL = np.zeros(num_iterations)
            CM_all = np.zeros((2, 2))
            CM_perc_all = np.zeros((2, 2))
            for r in range(0, num_iterations): #(nFolds-1)

                # display
                if log:
                    util.print_pers("", fileResultNameFull)
                    util.print_pers("Iteration n. {0}".format(r + 1), fileResultNameFull)
                    util.print_pers("", fileResultNameFull)

                # get current model
                if modelData['name'] == 'vgg16_bn_Mahdi':
                    # load pretrained
                    dirPretrainedModel = dirPretrainedModels + 'level_{}'.format(hierarNum+1) + '/' + modelData['name'] + '/'
                    fileSaveModel = open(os.path.join(dirPretrainedModel, 'model_1.dat'), 'rb')
                    currentModel = pickle.load(fileSaveModel)[0]
                    fileSaveModel.close()
                    currentModel.load_state_dict(torch.load(os.path.join(dirPretrainedModel, 'modelsave_1_final.pt')))
                    # block parameters
                    for param in currentModel.parameters():
                        param.requires_grad = True # deep tune
                    # change last layer
                    new_classifier = currentModel.classifierG
                    new_classifier[-1] = nn.Linear(modelData['sizeFeatures'], 2)
                    currentModel.classifierG = new_classifier
                    # image size
                    imageSize = 224

                if modelData['name'] == 'resnet18':
                    # load model
                    dirPretrainedModel = dirPretrainedModels + 'level_{}'.format(hierarNum+1) + '/' + modelData['name'] + '/'
                    fileSaveModel = open(os.path.join(dirPretrainedModel, 'model_1.dat'), 'rb')
                    currentModel = pickle.load(fileSaveModel)[0]
                    fileSaveModel.close()
                    currentModel.load_state_dict(torch.load(os.path.join(dirPretrainedModel, 'modelsave_1_final.pt')))
                    # block parameters
                    for param in currentModel.parameters():
                        param.requires_grad = True # deep tune
                    # change last layer
                    new_fc = nn.Linear(modelData['sizeFeatures'], 2)
                    currentModel.fc = new_fc
                    imageSize = 224

                #currentModel.double()
                # cuda
                if cuda:
                    currentModel.to('cuda')
                # log
                if log:
                    print(currentModel)

                # optim
                criterion = nn.CrossEntropyLoss()
                optimizer_ft = optim.SGD(currentModel.parameters(), lr=0.02, momentum=0.9, weight_decay=0.0005)
                # sched
                exp_lr_scheduler = list()
                exp_lr_scheduler.append(lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.5))

                # split into classes
                # first delete dir
                if os.path.exists(dirOutTrainTest):
                    shutil.rmtree(dirOutTrainTest)
                # create
                if not os.path.exists(dirOutTrainTest):
                    os.makedirs(dirOutTrainTest)
                split_folders.ratio(dirDbTest, output=dirOutTrainTest, seed=random(), ratio=(.4, .1, .5))
                util.print_pers("", fileResultNameFull)

                # preprocess
                transform = {
                    'train':
                        transforms.Compose([
                            transforms.CenterCrop(256),
                            transforms.Resize(imageSize, interpolation=PIL.Image.BILINEAR),
                            #transforms.RandomRotation(45),
                            transforms.ToTensor()
                        ]),
                    'val':
                        transforms.Compose([  # [1]
                            transforms.CenterCrop(256),
                            transforms.Resize(imageSize, interpolation=PIL.Image.BILINEAR),
                            transforms.ToTensor()
                        ])
                }


                # ------------------------------------------------------------------- TRAIN
                # load data
                # train
                all_idb2_train = datasets.ImageFolder(os.path.join(dirOutTrainTest, 'train'),
                                                           transform['train'])
                all_idb2_train_loader = torch.utils.data.DataLoader(all_idb2_train,
                                                                    batch_size=batch_sizeP_norm, shuffle=True,
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
                    transforms.CenterCrop(256),
                    transforms.Resize(imageSize, interpolation=PIL.Image.BILINEAR),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                    transforms.ToTensor(),
                    #transforms.Normalize(
                        #mean=[meanNorm, meanNorm, meanNorm],
                        #std=[stdNorm, stdNorm, stdNorm]),
                ])
                # val
                transform['val'] = transforms.Compose([
                    transforms.CenterCrop(256),
                    transforms.Resize(imageSize, interpolation=PIL.Image.BILINEAR),
                    #transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    #transforms.Normalize(
                        #mean=[meanNorm, meanNorm, meanNorm],
                        #std=[stdNorm, stdNorm, stdNorm]),
                ])
                print()

                # update data loaders
                # train
                all_idb2_train = datasets.ImageFolder(os.path.join(dirOutTrainTest, 'train'),
                                                           transform=transform['train'])
                all_idb2_train_loader = torch.utils.data.DataLoader(all_idb2_train,
                                                                    batch_size=batch_sizeP, shuffle=True,
                                                                    num_workers=numWorkersP, pin_memory=True)
                # val
                all_idb2_val = datasets.ImageFolder(os.path.join(dirOutTrainTest, 'val'),
                                                         transform=transform['val'])
                all_idb2_val_loader = torch.utils.data.DataLoader(all_idb2_val,
                                                                    batch_size=batch_sizeP, shuffle=False,
                                                                    num_workers=numWorkersP, pin_memory=True)

                # train
                util.print_pers("Training", fileResultNameFull)
                # train net
                currentModel = functions.train_model_val(currentModel, criterion,
                                                         optimizer_ft, exp_lr_scheduler,
                                                         num_epochs, dataset_sizes, all_idb2_train_loader, all_idb2_val_loader,
                                                         batch_sizeP, modelData['name'],
                                                         dirResult, r, fileResultNameFull, log, cuda)


                # visualize some outputs
                #functions.visualize_model(currentModel, all_idb2_val_loader, cuda, columnNames, num_images=6)
                #util.pause()


                # ------------------------------------------------------------------- TEST
                # torch.cuda.empty_cache()

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
                    transforms.CenterCrop(256),
                    transforms.Resize(imageSize, interpolation=PIL.Image.BILINEAR),
                    transforms.ToTensor(),
                    #transforms.Normalize(
                        #mean=[meanNorm, meanNorm, meanNorm],
                        #std=[stdNorm, stdNorm, stdNorm]),
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

                # loop on images
                # init
                predALL_test = torch.zeros(dataset_sizes['test'])
                labelsALL_test = torch.zeros(dataset_sizes['test'])
                for batch_num, (inputs, label) in enumerate(all_idb2_test_loader):

                    ##################
                    #if batch_num > 10:
                        #break
                    ##################

                    # get size of current batch
                    sizeCurrentBatch = label.size(0)

                    if batch_num % 100 == 0:
                        print("\t\tBatch n. {0} / {1}".format(batch_num, int(numBatches['test'])))

                    if plotta:
                        util.visImage(inputs)
                        util.print_pers("\tClasse: {0}".format(label), fileResultNameFull)
                        # util.pause()

                    # stack
                    indStart = batch_num * batch_sizeP
                    indEnd = indStart + sizeCurrentBatch

                    # extract features
                    if cuda:
                        inputs = inputs.to('cuda')
                        label = label.to('cuda')

                    # predict
                    with torch.set_grad_enabled(False):
                        outputs = currentModel(inputs)
                        if cuda:
                            outputs = outputs.to('cuda')

                        # softmax
                        _, preds = torch.max(outputs, 1)

                        predALL_test[indStart:indEnd] = preds
                        labelsALL_test[indStart:indEnd] = label

                # end for x,y

                # confusion matrix
                CM = confusion_matrix(labelsALL_test, predALL_test)
                CM_perc = CM / dataset_sizes['test'] # perc
                accuracyResult = util.accuracy(CM)
                CM_all = CM_all + CM
                CM_perc_all = CM_perc_all + CM_perc

                # print(output_test)
                util.print_pers("\tConfusion Matrix (%):", fileResultNameFull)
                util.print_pers("\t\t{0}".format(CM_perc * 100), fileResultNameFull)
                util.print_pers("\tAccuracy (%): {0:.2f}".format(accuracyResult * 100), fileResultNameFull)

                # assign
                accuracyALL[r] = accuracyResult

                # newline
                util.print_pers("", fileResultNameFull)

                # save iter
                fileSaveIter = open(os.path.join(dirResult, 'results_{0}.dat'.format(r+1)), 'wb')
                pickle.dump([accuracyResult], fileSaveIter)
                fileSaveIter.close()
                # fileSaveModelIter = open(os.path.join(dirResult, 'model_{0}.dat'.format(r+1)), 'wb')
                # pickle.dump([currentModel], fileSaveModelIter)
                # fileSaveModelIter.close()

                # del
                if cuda:
                    del currentModel
                    del all_idb2_train, all_idb2_train_loader
                    del all_idb2_val, all_idb2_val_loader
                    del all_idb2_test, all_idb2_test_loader
                    del inputs, label
                    del outputs, preds
                    del criterion, optimizer_ft, exp_lr_scheduler
                    torch.cuda.empty_cache()

            # end loop on iterations

            # average accuracy
            meanAccuracy = np.mean(accuracyALL)
            stdAccuracy = np.std(accuracyALL)
            meanCM = CM_all / num_iterations
            meanCM_perc = CM_perc_all / num_iterations

            # display
            util.print_pers("", fileResultNameFull)
            util.print_pers("Mean classification accuracy over {0} iterations (%); {1:.2f}".format(num_iterations, meanAccuracy * 100),
                            fileResultNameFull)
            util.print_pers("Std classification accuracy over {0} iterations (%); {1:.2f}".format(num_iterations, stdAccuracy * 100),
                            fileResultNameFull)
            util.print_pers("\tMean Confusion Matrix over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0}".format(meanCM_perc * 100), fileResultNameFull)
            util.print_pers("\tTP (mean) over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0:.2f}".format(meanCM_perc[1, 1] * 100), fileResultNameFull)
            util.print_pers("\tTN (mean) over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0:.2f}".format(meanCM_perc[0, 0] * 100), fileResultNameFull)
            util.print_pers("\tFP (mean) over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0:.2f}".format(meanCM_perc[0, 1] * 100), fileResultNameFull)
            util.print_pers("\tFN (mean) over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0:.2f}".format(meanCM_perc[1, 0] * 100), fileResultNameFull)

            #close
            fileResult.close()

            # save
            fileSaveFinal = open(os.path.join(dirResult, 'resultsFinal.dat'), 'wb')
            pickle.dump([meanAccuracy], fileSaveFinal)
            fileSaveFinal.close()

            # del
            torch.cuda.empty_cache()




