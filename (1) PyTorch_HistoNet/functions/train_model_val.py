import torch
import time
import copy
import numpy as np
import os
import torch.nn as nn
from util import pause
from util import getClassCount
from util import normImageCustom
from util import imshow
from util import visImage
from util import print_pers


# multi label accuracy
def acc_fun_geno(preds, labelsTens):
    labels_all = labelsTens.data.int()
    runningsum = 0
    preds_cpu = preds.cpu()
    labels_all_cpu = labels_all.cpu()
    for i, pred_sample in enumerate(preds_cpu):
        labelv = labels_all_cpu[i]
        numerator = torch.sum(np.bitwise_and(pred_sample, labelv))
        denominator = torch.sum(np.bitwise_or(pred_sample, labelv))
        runningsum += (numerator.double()/denominator.double())
    return runningsum


# training with validation
def train_model_val(model, classVec,
                    optimizer, scheduler,
                    num_epochs, dataset_sizes, dataloader_train, dataloader_val,
                    batch_sizeP, num_classes, modelName,
                    dirResults, iteration, fileResultNameFull, log, cuda):

    # check if final already exists
    fileNameSaveFinal = 'modelsave_{0}_final.pt'.format(iteration+1)
    if os.path.isfile(os.path.join(dirResults, fileNameSaveFinal)):
        # display
        if log:
            print_pers('\tModel loaded', fileResultNameFull)
        model.load_state_dict(torch.load(os.path.join(dirResults, fileNameSaveFinal)))
        return model

    #init time
    since = time.time()

    # init best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # compute num batches
    numBatches = {}
    numBatches['train'] = np.round(dataset_sizes['train'] / batch_sizeP)
    numBatches['val'] = np.round(dataset_sizes['val'] / batch_sizeP)

    # class weights computed on train and val together
    datasetSizeAll = dataset_sizes['train'] + dataset_sizes['val']
    classCountAll = getClassCount(classVec['train']) + getClassCount(classVec['val'])

    # if no count, put 1
    numSub = 10
    for listc, tt in enumerate(classCountAll):
        if tt < numSub:
            classCountAll[listc] = numSub

    weightsBCE = torch.FloatTensor(datasetSizeAll / classCountAll)
    if cuda:
        weightsBCE = weightsBCE.to('cuda')

    #print(classCountAll)
    #print(weightsBCE)

    # check if partial results
    entries = os.listdir(dirResults)
    max_epoch = -1
    for entry in entries:
        if entry.endswith(".pt"):
            entry2 = os.path.splitext(entry)
            temp = entry2[0].split('_')
            # onlyy for this iteration
            if int(temp[1]) != (iteration+1):
                continue
            entry3 = temp[-1]
            saved_epoch = int(entry3)
            if saved_epoch > max_epoch:
                max_epoch = saved_epoch

    if max_epoch > -1:
        fileNameSave = 'modelsave_{0}_epoch_{1}.pt'.format(iteration+1, max_epoch)
        model.load_state_dict(torch.load(os.path.join(dirResults, fileNameSave)))

    # loop on epochs
    for epoch in range(num_epochs):

        # continue from saved epoch
        if epoch <= max_epoch:
            continue

        # display
        if log:
            print_pers('\tEpoch {}/{}'.format(epoch+1, num_epochs), fileResultNameFull)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # init losses and corrects
            running_loss = 0.0
            running_corrects = 0.0
            running_corrects2 = 0.0

            # choose dataloader
            if phase == 'train':
                dataloaders_chosen = dataloader_train
            if phase == 'val':
                dataloaders_chosen = dataloader_val

            # Iterate over data.
            for batch_num, (inputs, dummyTargets, filename, label) in enumerate(dataloaders_chosen):

                # get size of current batch
                sizeCurrentBatch = dummyTargets.size(0)

                ##################
                #if batch_num > 10:
                    #break
                ##################

                # cuda
                if cuda:
                    inputs = inputs.to('cuda')
                    label = label.to('cuda')

                # display
                if batch_num % 100 == 0:
                    print_pers("\t\tBatch n. {0} / {1}".format(batch_num, int(numBatches[phase])), fileResultNameFull)

                # indexes
                #indStart = batch_num * batch_sizeP
                #indEnd = indStart + sizeCurrentBatch

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if cuda:
                        outputs = outputs.to('cuda')

                    m = nn.Sigmoid()
                    preds = (m(outputs) > 0.5).int()

                    #print(label)
                    #print(label.size())
                    #print(outputs)
                    #print(outputs.size())

                    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weightsBCE)
                    criterion = torch.nn.MultiLabelSoftMarginLoss(weight=weightsBCE)
                    loss = criterion(outputs.float(), label.float())

                    #print(loss)
                    #pause()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                with torch.no_grad():
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == label.data.int())
                    running_corrects2 += acc_fun_geno(preds, label)

            # update schedulers
            if phase == 'train':
                for schedulerSingle in scheduler:
                    schedulerSingle.step()

            # compute epochs losses
            with torch.no_grad():
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / (dataset_sizes[phase] * num_classes)
                epoch_acc2 = running_corrects2.double() / dataset_sizes[phase]

            # display
            if log:
                print_pers('\t\t{} Loss: {:.4f} Acc (1-HL): {:.4f}; (MultiL): {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_acc2),
                           fileResultNameFull)

            # if greater val accuracy, deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # save model at epoch
        if epoch % 20 == 0:
            fileNameSave = 'modelsave_{0}_epoch_{1}.pt'.format(iteration+1, epoch)
            torch.save(model.state_dict(), os.path.join(dirResults, fileNameSave))

        # del
        del inputs, dummyTargets, label
        torch.cuda.empty_cache()

    # time
    time_elapsed = time.time() - since
    print('\tTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('\tBest val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # save final
    torch.save(model.state_dict(), os.path.join(dirResults, fileNameSaveFinal))

    # del
    torch.cuda.empty_cache()

    return model