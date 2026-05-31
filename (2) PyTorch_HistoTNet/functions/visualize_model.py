import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import util.imshow as imshow
import numpy as np

def visualize_model(model, dataloaders, cuda, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(1)

    with torch.no_grad():
        for i, (inputs, dummyTarget, filename, labels) in enumerate(dataloaders):
            if cuda:
                inputs = inputs.to('cuda')

            outputs = model(inputs)
            m = nn.Sigmoid()
            #preds = torch.round(m(outputs)).int()
            preds = (m(outputs) > 0.5).int()

            for j in range(inputs.size()[0]):
                labelJ = labels[j].numpy()
                predJ = preds[j].cpu().numpy()
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('real: {}; predicted: {}'.format(labelJ, predJ))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.show()
                    return

                plt.show()
        model.train(mode=was_training)