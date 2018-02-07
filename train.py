import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import EvolvingModel as Model
from torch.autograd import Variable
from EvolvingModel import BasicBlock, Bottleneck

import matplotlib.pyplot as plt

from skorch.net import NeuralNetClassifier

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor

#=============== Hyperparameters =================#

TASK_NUMBER = 1 #Current task Starting from 1
MODULAR = True

pooling_num = int(28 / ((TASK_NUMBER - 1)*2)) if TASK_NUMBER > 1 else 28

start_epoch = 0
epochs = 120
momentum = 0.9
weight_decay = 1e-4
learning_rate = 0.1
batch_size = 256
print_freq = 20
plot_every = 5

#=================================================#

prev_model_loader = "Module_{}.pt".format(TASK_NUMBER-1)
path = './Task_{}/'.format(TASK_NUMBER)

#=================================================#

#The following lists are defined here to be treated as global objects. Their purpose is to be plotted

train_losses = []
test_losses = []

train_accuracies = []
test_accuracies = []

train_precision = []
test_precision = []

train_recall = []
test_recall = []


def main():


    # ======= ====== ======= =======
    # ===== Defining The Model =====
    print("Using model resnet18")
    blocklist = [2,2]
    stride = [1,2]
    block_type = [BasicBlock, BasicBlock] #the length of these lists is the number of residual blocks, the first list defines how many layers each block will have and the second the number of strides and the last one which type of block to be used
    for i in range(TASK_NUMBER-1):
        blocklist += [2]
        stride += [2]
        block_type += [BasicBlock]


    model = Model.ModNet(blocklist=blocklist, stride=stride, pooling_num=pooling_num, block=block_type, num_classes=1)

    # define loss function (criterion) and optimizer
    """
    This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
    """
    criterion = nn.BCEWithLogitsLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                                momentum=momentum, weight_decay=weight_decay)
    if TASK_NUMBER > 1 and MODULAR:
        # original saved file with DataParallel
        checkpoint = torch.load(prev_model_loader)['state_dict']
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        trained_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # remove `module.`
            trained_state_dict[name] = v
        # load params
        #The fully connected is discarded
        del trained_state_dict['fc.weight']
        del trained_state_dict['fc.bias']
        model.load_state_dict(trained_state_dict, strict=False) #We only want to match part of the model
        print("Succesfully Loaded Last Model")
        model.freeze(trained_state_dict)


    #model = Model.One_More_Module(model.stacked_layers)

    #model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936

    # ====== ===== ====== ======
    # ====== Loading Data ======
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')
    #normalize to remove all intensity values from the image while preserving color values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    #print(train_dataset.shape)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)


    # ====== ===== ====== ===== ======
    # ===== ===== Training ===== =====
    start = time.clock()
    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on test set
        validate(val_loader, model, criterion, epoch)

    print("Total training time for {} epochs was {}".format(epochs, time.clock()-start))
    torch.save({
    'epoch': epoch + 1,
    'state_dict': model.cpu().state_dict(),
    'optimizer' : optimizer.state_dict()
    }, 'Module_{}.pt'.format(TASK_NUMBER))

    # ===== ===== ===== ===== #
    # ======  Plotting ====== #
    plt.subplot(221)
    plt.plot(list(range(start_epoch,epochs, plot_every)), train_losses, label = "Training Loss")
    plt.plot(list(range(start_epoch,epochs, plot_every)), test_losses, label = "Test Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(list(range(start_epoch, epochs, plot_every*2)))

    plt.subplot(222)
    plt.plot(list(range(start_epoch,epochs, plot_every)), train_accuracies, label = "Training Accuracy")
    plt.plot(list(range(start_epoch,epochs, plot_every)), test_accuracies, label = "Test Accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(list(range(start_epoch, epochs, plot_every*2)))
    """
    plt.subplot(223)
    plt.plot(list(range(start_epoch,epochs, plot_every)), test_precision, label = "Test Precision")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.xticks(list(range(start_epoch, epochs, plot_every*2)))

    plt.subplot(224)
    plt.plot(list(range(start_epoch,epochs, plot_every)), test_recall, label = "Test Recall")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.xticks(list(range(start_epoch, epochs, plot_every*2)))

    """
    if MODULAR:
        plt.savefig("ModularNet_Training_Task_{}".format(TASK_NUMBER))
    else:
        plt.savefig("SimpleNet_Training_Task_{}".format(TASK_NUMBER))


    # ====== ====== ====== ====== #

mean = lambda x : sum(x)/len(x)


def train(train_loader, model, criterion, optimizer, epoch, accuracies = None):

    # switch to train mode
    model.train()

    accuracies = []
    losses = []

    for i, (input, target) in enumerate(train_loader):

        target = target.type(dtype).unsqueeze(0).t().cuda(async=True)
        input_var = Variable(input.type(dtype))
        target_var = Variable(target)

        # Compute output
        output = model.forward(x=input_var)
        loss = criterion(output, target_var)
        losses.append(loss.data[0])

        # Compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics. To get the predicted probabilities we need to add a Sigmoid layer
        predicted_probabilities = nn.Sigmoid()(output)
        metrics = Metrics(predicted_probabilities, target_var)

        accuracies.append(metrics.accuracy())

        if i % print_freq == 0:
            print('Epoch: [{}][{}/{}]\t Loss {} (Average {})\t'.format(
                   epoch, i, len(train_loader), losses[i], mean(losses)))

    if epoch % plot_every == 0:
        train_accuracies.append(mean(accuracies))
        train_losses.append(mean(losses))



def validate(val_loader, model, criterion, epoch, accuracies = None, precision = None, recall = None):

    # switch to evaluate mode
    model.eval()

    accuracies = []
    losses = []

    for i, (input, target) in enumerate(val_loader):

        target = target.type(dtype).unsqueeze(0).t().cuda(async=True)
        input_var = Variable(input.type(dtype), volatile=True)
        target_var = Variable(target, volatile=True)
        NeuralNetClassifier
        # Compute output
        output = model.forward(input_var)
        loss = criterion(output, target_var)
        losses.append(loss.data[0])

        # Calculate metrics. To get the predicted probabilities we need to add a Sigmoid layer
        predicted_probabilities = nn.Sigmoid()(output)
        metrics = Metrics(predicted_probabilities, target_var)
        accuracies.append(metrics.accuracy())

    print('Test: Loss Average {}\t'.format(mean(losses)))
    if epoch % plot_every == 0:
        test_accuracies.append(mean(accuracies))
        test_losses.append(mean(losses))

class Metrics(object):

    def __init__(self, predicted_probabilities, target):
        from sklearn import metrics
        self.predicted_probabilities = predicted_probabilities.data.cpu()
        self.target = target.data.cpu()
        self.predictions = torch.zeros(self.predicted_probabilities.size())
        self.predictions[self.predicted_probabilities > 0.5] = 1 #where the probability is higher than 0.5, label 1 is assigned
        # save confusion matrix and slice into four pieces
        """
        confusion = metrics.confusion_matrix(y_test, y_pred_class)
        print(confusion)
        #[row, column]
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        """

    def accuracy(self):
        correct = (self.predictions == self.target).sum()
        accuracy = correct/self.predictions.size()[0]
        return accuracy


def adjust_learning_rate(optimizer, epoch, decay_rate=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (decay_rate ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
"""
"Before executing the code, python will define a few special variables. For example, if the python interpreter is running that module (the source file) as the main program, it sets the special __name__ variable to have a value "__main__". If this file is being imported from another module, __name__ will be set to the module's name."
"""
