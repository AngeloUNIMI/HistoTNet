import numpy as np
import torch
from util import pause

def computeMeanStd_RGB(dataloader, dataset_sizes, batch_sizeP, cuda):

    numBatches = np.round(dataset_sizes / batch_sizeP)

    pop_mean_R = []
    pop_mean_G = []
    pop_mean_B = []
    pop_std0_R = []
    pop_std0_G = []
    pop_std0_B = []

    for i, (data, y) in enumerate(dataloader):

        # display
        if i % 100 == 0:
            print("\tBatch n. {0} / {1}".format(i, int(numBatches)))

        if cuda:
            data = data.to('cuda')

        # shape (3,)
        batch_mean_R = torch.mean(data[:,0,:])
        batch_mean_G = torch.mean(data[:,1,:])
        batch_mean_B = torch.mean(data[:,2,:])

        batch_std0_R = torch.std(data[:,0,:])
        batch_std0_G = torch.std(data[:,1,:])
        batch_std0_B = torch.std(data[:,2,:])

        if cuda:
            batch_mean_R = batch_mean_R.detach().to('cpu')
            batch_mean_G = batch_mean_G.detach().to('cpu')
            batch_mean_B = batch_mean_B.detach().to('cpu')
            batch_std0_R = batch_std0_R.detach().to('cpu')
            batch_std0_G = batch_std0_G.detach().to('cpu')
            batch_std0_B = batch_std0_B.detach().to('cpu')

        pop_mean_R.append(batch_mean_R)
        pop_mean_G.append(batch_mean_G)
        pop_mean_B.append(batch_mean_B)
        pop_std0_R.append(batch_std0_R)
        pop_std0_G.append(batch_std0_G)
        pop_std0_B.append(batch_std0_B)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean_R = np.mean(pop_mean_R)
    pop_mean_G = np.mean(pop_mean_G)
    pop_mean_B = np.mean(pop_mean_B)
    pop_std0_R = np.mean(pop_std0_R)
    pop_std0_G = np.mean(pop_std0_G)
    pop_std0_B = np.mean(pop_std0_B)

    return pop_mean_R, pop_mean_G, pop_mean_B, pop_std0_R, pop_std0_G, pop_std0_B


def computeMeanStd(dataloader_all, dataset_sizes, batch_sizeP, cuda):

    numBatches = np.round(dataset_sizes / batch_sizeP)

    pop_mean = []
    pop_std0 = []

    for dataloader in dataloader_all:

        for i, (data, y) in enumerate(dataloader):

            ##################
            #if i > 0:
                #break
            ##################

            #print(data.size())
            #pause()

            # display
            #if i % 100 == 0:
                #print("\tBatch n. {0} / {1}".format(i, int(numBatches)))

            if cuda:
                data = data.to('cuda')

            # shape (3,)
            batch_mean = torch.mean(data)
            batch_std0 = torch.std(data)

            if cuda:
                batch_mean = batch_mean.detach().to('cpu')
                batch_std0 = batch_std0.detach().to('cpu')

            pop_mean.append(batch_mean)
            pop_std0.append(batch_std0)

            #print(pop_mean)
            #print(len(pop_mean))
            #pause()

            #if i > 100:
            #break

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.mean(pop_mean)
    pop_std0 = np.mean(pop_std0)

    return pop_mean, pop_std0
