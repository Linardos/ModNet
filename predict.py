import os
import time
import pickle
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

TASK_TARGET = [1, 'Animals']
path_to_task = './Task_{}_{}/'.format(TASK_TARGET[0], TASK_TARGET[1])
prev_model_loader = "Module_1_Animals.pt"
# ======= ====== ======= =======
# ===== Defining The Model =====

pooling_num = int(28 / ((TASK_TARGET[0] - 1)*2)) if TASK_TARGET[0] > 1 else 28

blocklist = [2,2]
stride = [1,2]
block_type = [BasicBlock, BasicBlock] #the length of these lists is the number of residual blocks, the first list defines how many layers each block will have and the second the number of strides and the last one which type of block to be used
for i in range(TASK_TARGET[0]-1):
    blocklist += [2]
    stride += [2]
    block_type += [BasicBlock]


model = Model.ModNet(blocklist=blocklist, stride=stride, pooling_num=pooling_num, block=block_type, num_classes=1)


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

model.load_state_dict(trained_state_dict, strict=False) #We only want to match part of the model
print("Succesfully Loaded Pre-Trained Layers")

dataset =
