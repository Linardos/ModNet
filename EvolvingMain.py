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
import OneModel as Model

import matplotlib.pyplot as plt

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor

#=============== Hyperparameters =================#

TASK_NUMBER = 2 #Current task Starting from 1
MODULAR = True

pooling_num = int(28 / ((TASK_NUMBER - 1)*2)) if TASK_NUMBER > 1 else 28

start_epoch = 0
epochs = 5
weight_decay = 1e-4
learning_rate = 0.1
batch_size = 256
print_freq = 20
plot_every = 5

#=================================================#

prev_model_loader = "Module_{}.pt".format(TASK_NUMBER-1)
path = './Task_{}/'.format(TASK_NUMBER)

def main():


    # ======= ====== ======= =======
    # ===== Defining The Model =====
    print("Using model resnet18")
    blocklist = [2,2]
    stride = [1,2] #the length of these lists is the number of residual blocks, the first list defines how many layers each block will have and the second the number of strides
    for i in range(TASK_NUMBER-1):
        blocklist.append(2)
        stride.append(2)
    model = Model.ModNet(blocklist=blocklist, stride=stride, pooling_num=pooling_num, num_classes=1)

    # define loss function (criterion) and optimizer
    criterion = nn.BCEWithLogitsLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), learning_rate,
                                    weight_decay=weight_decay)
    if TASK_NUMBER > 1 and MODULAR:
        # original saved file with DataParallel
        checkpoint = torch.load(prev_model_loader)['state_dict']
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        #The fully connected is discarded
        del new_state_dict['fc.weight']
        del new_state_dict['fc.bias']
        model.load_state_dict(new_state_dict, strict=False) #We only want to match part of the model
        model.freeze(new_state_dict)

    print("Succesfully Loaded Last Model")

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
    # ====== Lists For Plotting ======

    train_losses = []
    test_losses = []

    train_accuracies = []
    test_accuracies = []

    train_precision = []
    test_precision = []

    train_recall = []
    test_recall = []

    # ====== ===== ====== ===== ======
    # ===== ===== Training ===== =====

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        losses = AverageMeter() #This is a class used to keep track of losses in current epoch, in order to print and plot
        accuracies = AverageMeter()
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, losses, accuracies)
        #Save to plotting list
        if epoch % plot_every == 0:
            train_losses.append(losses.avg)
            train_accuracies.append(accuracies.avg)

        # evaluate on test set
        losses = AverageMeter()
        accuracies = AverageMeter()
        precision = AverageMeter()
        recall = AverageMeter()
        validate(val_loader, model, criterion, losses, accuracies, precision, recall)
        if epoch % plot_every == 0:
            test_losses.append(losses.avg)
            test_accuracies.append(accuracies.avg)
            test_precision.append(precision.avg)
            test_recall.append(recall.avg)

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

    if MODULAR:
        plt.savefig("ModularNet Training Task {}".format(TASK_NUMBER))
    else:
        plt.savefig("SimpleNet Training Task {}".format(TASK_NUMBER))


    # ====== ====== ====== ====== #



def train(train_loader, model, criterion, optimizer, epoch, losses = None, accuracies = None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.type(dtype).unsqueeze(0).t().cuda(async=True)
        input_var = torch.autograd.Variable(input.type(dtype))
        target_var = torch.autograd.Variable(target)
        # compute output
        #print(input_var.size())
        output = model.forward(x=input_var)
        loss = criterion(output, target_var)

        losses.update(loss.data[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #==== Accuracy Calculation ===#
        prob_vector = nn.Sigmoid()(output.cpu()) #Get the probabilities from the final layer
        predictions = torch.zeros(prob_vector.size())
        predictions[prob_vector.data > 0.5] = 1 #where the probability is higher than 0.5, label 1 is assigned
        correct = (predictions == target.cpu()).sum()
        accuracy = correct/predictions.size()[0]
        accuracies.update(accuracy)

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))




def validate(val_loader, model, criterion, losses = None, accuracies = None, precision = None, recall = None):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()
    correct = 0
    for i, (input, target) in enumerate(val_loader):
        start = time.time()

        target = target.type(dtype).unsqueeze(0).t().cuda(async=True)
        input_var = torch.autograd.Variable(input.type(dtype), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        #print("input %f"%input_var.data)
        # compute output
        output = model.forward(input_var)
        loss = criterion(output, target_var)

        losses.update(loss.data[0])

        #correct += (target == labels).sum()
        # measure elapsed time
        batch_time.update(time.time() - start)

        CPU_target = target.cpu()
        #==== Accuracy Calculation ===#
        prob_vector = nn.Sigmoid()(output.cpu()) #Get the probabilities from the final layer
        predictions = torch.zeros(prob_vector.size())
        predictions[prob_vector.data > 0.5] = 1
        correct = (predictions == CPU_target).sum()
        accuracy = correct/predictions.size()[0]
        accuracies.update(accuracy)

        #==== Precision Calculation ===#
        positives = np.where(predictions == 1)[0] #get the indices where predictions are 1
        TruePositives_1 = (predictions.numpy()[positives] == CPU_target.numpy()[positives]).sum()

        precision.update(TruePositives_1/len(positives))

        relevant = np.where(CPU_target == 1)[0]
        TruePositives_2 = (predictions.numpy()[relevant] == CPU_target.numpy()[relevant]).sum()

        recall.update(TruePositives_2/len(relevant))

        assert(TruePositives_1 == TruePositives_2) #2 different ways to calculate true positives were used, to validate that everything is correct

    print('Average Precision is {} \n'.format(precision.avg))

    print('Test: '
      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
      'Loss Average {loss.avg:.4f}\t'.format(batch_time=batch_time, loss=losses))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
"""
"Before executing the code, python will define a few special variables. For example, if the python interpreter is running that module (the source file) as the main program, it sets the special __name__ variable to have a value "__main__". If this file is being imported from another module, __name__ will be set to the module's name."
"""
