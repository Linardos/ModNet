import torch.nn as nn
import math

print('Succesfully imported Model module')

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


from collections import OrderedDict
class LastUpdatedOrderedDict(OrderedDict):
    'Store items in the order the keys were last added'

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, projection=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.projection = projection
        self.stride = stride

    def forward(self, x):
        #print(x.size())
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.projection is not None:
            residual = self.projection(x) #projection is used to match the dimension of the residual to the matrix to which it is added

        out += residual
        out = self.relu(out)
        #print(out.size())

        return out


#When changing tasks it would make sense to move through a bottleneck in order to reduce redundant information from the previous task.
class Bottleneck(nn.Module):
    #changed from 4 to 2
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, projection=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.projection = projection
        self.stride = stride

    def forward(self, x):
        residual = x
        #print('Sizes while passing through the network: \n')
        #print(x.size())
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print(out.size())

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #print(out.size())

        out = self.conv3(out)
        out = self.bn3(out)
        #print(out.size())

        if self.projection is not None:
            residual = self.projection(x)

        out += residual
        out = self.relu(out)
        #print(out.size())
        import time
        time.sleep(60)

        return out

class ModNet(nn.Module):
    #use num_blocks to for loop dictionary creation in order to automate layer making
    def __init__(self, blocklist, num_classes=1, stride = [1,2,2], block=BasicBlock):
        #blocklist specifies the number of layers in each block
        self.inplanes = 64
        super(ModNet, self).__init__()

        #This structure has the advantage of being able to set the number of layers as a hyperparameter. The hyperparameters in this case are lists of same length which equals the number of blocks we want to use and contain how many layers each block has and the stride movement
        layers = LastUpdatedOrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ("layer1", self._make_block(block, 64, blocklist[0], stride=stride[0]))
                ])
        p = 128
        for i in range(1,len(blocklist)):
            layer = self._make_block(block, p, blocklist[i], stride=stride[i])
            layers.__setitem__("layer{}".format(i+1), layer)
            p = 2*p

        self.stacked_layers = nn.Sequential(layers)
        self.avgpool = nn.AvgPool2d(14, stride=1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        #Weight Initialization (Supposedly from : Delving deep into rectifiers: Surpassing human-level performance on imagenet classification.  In ICCV , 2015.)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): #if m is an instance of class Conv2d
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_block(self, block, planes, blocklist, stride=1):
        projection = None #projection is used to match the dimension of the residual to the matrix to which it is added
        if stride != 1 or self.inplanes != planes * block.expansion:
            projection = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, projection))
        self.inplanes = planes * block.expansion
        for i in range(1, blocklist):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, Classify=True):

        x = self.stacked_layers(x)

        if Classify == True:

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

        else:

            return x



from torch.nn.parallel import data_parallel

class One_More_Module(nn.Module): #Maybe use as input the trained module?

    def __init__(self, pretrained_model, layers=2, num_classes=1, block=[Bottleneck, BasicBlock]):
        self.inplanes = 256 #changed from 64 This is the number of channels the makeblock expects
        super(One_More_Module, self).__init__()

        self.trained_conv = pretrained_model.stacked_layers
        self.next_layer = self._make_block(block[0], 512, layers, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block[0].expansion, num_classes)
        for m in self.next_layer.modules():
            if isinstance(m, nn.Conv2d): #if m is an instance of class Conv2d
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for param in self.trained_conv.parameters():
            param.requires_grad = False

    def _make_block(self, block, planes, blocklist, stride=1):
        projection = None
        #print(planes*block.expansion)
        if stride != 1 or self.inplanes != planes * block.expansion:
            projection = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, projection))
        self.inplanes = planes * block.expansion #increase the number of channels relative to block expansion
        for i in range(1, blocklist):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, Classify=True):
        #print('pre-block')
        #print(x.size())
        #print('pre-block')
        x = self.trained_conv(x)
        #x = data_parallel(self.next_layer, x)
        x = self.next_layer(x)
        #print('after-block')
        #print(x.size())
        #print('after-block')

        if Classify == True:

            x = self.avgpool(x)
            #print(x.size())
            x = x.view(x.size(0), -1)
            #print(x.size())
            x = self.fc(x)

            return x

        else:

            return x
