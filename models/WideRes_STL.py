import os, sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

# def conv_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         init.xavier_uniform(m.weight, gain=np.sqrt(2))
#         init.constant(m.bias, 0)
#     elif classname.find('BatchNorm') != -1:
#         init.constant(m.weight, 1)
#         init.constant(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm='bn'):
        super(wide_basic, self).__init__()
        if norm=='bn': self.bn1 = nn.BatchNorm2d(in_planes)
        elif norm=='in': self.bn1 = nn.InstanceNorm2d(in_planes)
#         self.film1 = film(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        
        if norm=='bn': self.bn2 = nn.BatchNorm2d(planes)
        elif norm=='in': self.bn2 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x, factor=None, bias=None):
        out1 = self.conv1(F.relu(self.bn1(x)))
        out2 = self.conv2(F.relu(self.bn2(out1)))
        out2 += self.shortcut(x.contiguous())

        return out2


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, fc, norm='bn'):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        self.n = int((depth - 4) / 6) # num blocks
        k = widen_factor
        filter = [16, 16 * k, 32 * k, 64 * k]

        # input conv
#         self.film = film(filter[0])
        if norm=='bn': self.bn = nn.BatchNorm2d(filter[0])
        elif norm=='in': self.bn = nn.InstanceNorm2d(filter[0])
        self.conv1 = conv3x3(3, filter[0], stride=1)
        
        # layer1
        depth = self.in_planes
        widen_factor = filter[1]
        stride = 2
        strides = [stride] + [1] * (self.n - 1)
        self.layer1 = nn.ModuleList([wide_basic(self.in_planes, filter[1], strides[0])])
        for i in range(1, self.n):
            self.layer1.append(wide_basic(filter[1], filter[1], strides[i]))
        
        # layer2
        stride = 2
        strides = [stride] + [1] * (self.n - 1)
        self.layer2 = nn.ModuleList([wide_basic(filter[1], filter[2], strides[0])])
        for i in range(1, self.n):
            self.layer2.append(wide_basic(filter[2], filter[2], strides[i]))
            
        # layer3
        stride = 2
        strides = [stride] + [1] * (self.n - 1)
        self.layer3 = nn.ModuleList([wide_basic(filter[2], filter[3], strides[0])])
        for i in range(1, self.n):
            self.layer3.append(wide_basic(filter[3], filter[3], strides[i]))
        
#         self.bn1 = nn.BatchNorm2d(filter[3], momentum=0.9)
#         self.film_last = film(filter[3])
        if norm=='bn': self.bn_last = nn.BatchNorm2d(filter[3])
        elif norm=='in': self.bn_last = nn.InstanceNorm2d(filter[3])
        
        # output layer
        self.linear = nn.ModuleList([nn.Sequential(
            nn.Linear(filter[3], num_classes[0]),
            nn.Softmax(dim=1))])
        
        for j in range(1, len(num_classes)):
            self.linear.append(nn.Sequential(nn.Linear(filter[3], num_classes[j]),
                                                 nn.Softmax(dim=1)))


    def forward(self, x, k, task_vec=None):
        
        g_encoder = [0] * 4
        # input conv
        param_idx = 0
        g_encoder[0] = self.bn(self.conv1(x))
        
        # layer1
        g_encoder[1] = self.layer1[0](g_encoder[0])
        g_encoder[1] = self.layer1[1](g_encoder[1])
        g_encoder[1] = self.layer1[2](g_encoder[1])
        g_encoder[1] = self.layer1[3](g_encoder[1])
                
        # layer2   
        g_encoder[2] = self.layer2[0](g_encoder[1])
        g_encoder[2] = self.layer2[1](g_encoder[2])
        g_encoder[2] = self.layer2[2](g_encoder[2])
        g_encoder[2] = self.layer2[3](g_encoder[2])
        
    
        # layer3 
        g_encoder[3] = self.layer3[0](g_encoder[2])
        g_encoder[3] = self.layer3[1](g_encoder[3])
        g_encoder[3] = self.layer3[2](g_encoder[3])
        g_encoder[3] = self.layer3[3](g_encoder[3])
                
        g_encoder[3] = F.relu(self.bn_last(g_encoder[3]))

        pred = F.avg_pool2d(g_encoder[-1], 8)
        pred = pred.contiguous().view(pred.size(0), -1)

        out = self.linear[k](pred)
        return out

    def model_fit(self, x_pred, x_output, num_output, device):
        # convert a single label into a one-hot vector
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        # apply cross-entropy loss
        loss = x_output_onehot * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)