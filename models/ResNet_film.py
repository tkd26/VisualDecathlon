# models.py
# created by Sylvestre-Alvise Rebuffi [srebuffi@robots.ox.ac.uk]
# Copyright © The University of Oxford, 2017-2020
# This code is made available under the Apache v2.0 licence, see LICENSE.txt for details
# resadapterの論文から引用
# https://github.com/srebuffi/residual_adapters

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

# import config_task
import math
import os, sys
import numpy as np

class film_generator(nn.Module):
    def __init__(self, norm_nc, cond_nc, fc=1, fc_nc=64):
        super().__init__()
        
        self.fc = fc
        self.relu = torch.nn.ReLU()
        if self.fc==1:
            self.transform = nn.Linear(cond_nc, norm_nc*2)
        if self.fc==3:
            self.transform1 = nn.Linear(cond_nc, fc_nc)
            self.transform2 = nn.Linear(fc_nc, fc_nc)
            self.transform = nn.Linear(fc_nc, norm_nc*2)
        if self.fc==5:
            self.transform1 = nn.Linear(cond_nc, fc_nc)
            self.transform2 = nn.Linear(fc_nc, fc_nc)
            self.transform3 = nn.Linear(fc_nc, fc_nc)
            self.transform4 = nn.Linear(fc_nc, fc_nc)
            self.transform = nn.Linear(fc_nc, norm_nc*2)
        self.transform.bias.data[:norm_nc] = 1
        self.transform.bias.data[norm_nc:] = 0
        
    def forward(self, cond):
        if self.fc==1:
            param = self.transform(cond).unsqueeze(2).unsqueeze(3)
        if self.fc==3:
            param = self.relu(self.transform1(cond))
            param = self.relu(self.transform2(param))
            param = self.transform(param).unsqueeze(2).unsqueeze(3)
        if self.fc==5:
            param = self.relu(self.transform1(cond))
            param = self.relu(self.transform2(param))
            param = self.relu(self.transform3(param))
            param = self.relu(self.transform4(param))
            param = self.transform(param).unsqueeze(2).unsqueeze(3)

        factor, bias = param.chunk(2, 1)
        return factor, bias
    
class film(nn.Module):
    def __init__(self, norm_nc, mode_norm='bn', version_film='film'):
        super().__init__()
        '''
        ここでbatchかinstanceかを変える
        film, no_film, binary
        '''
        self.version_film = version_film
        if self.version_film=='film' or self.version_film=='binary':
            if mode_norm=='bn':
                self.norm = nn.BatchNorm2d(norm_nc, affine=False)
            elif mode_norm=='in':
                self.norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif self.version_film=='no_film':
            self.norm = nn.BatchNorm2d(norm_nc, affine=True)
            
    def forward(self, x, factor, bias, visualize=False, k=None):
        normalized = self.norm(x)

        if self.version_film == 'film':
            out = normalized * factor + bias 
        elif self.version_film=='binary':
            factor_binary = torch.where(factor > 0, torch.ones(1).cuda(), torch.zeros(1).cuda())
            out = normalized * factor + bias 
        elif self.version_film=='no_film':
            out = normalized
        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1_fonc(in_planes, out_planes=None, stride=1, bias=False):
    if out_planes is None:
        return nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class conv1x1(nn.Module):
    
    def __init__(self, planes, out_planes=None, stride=1):
        super(conv1x1, self).__init__()
        # if config_task.mode == 'series_adapters':
        #     self.conv = nn.Sequential(nn.BatchNorm2d(planes), conv1x1_fonc(planes))
        # elif config_task.mode == 'parallel_adapters':
        #     self.conv = conv1x1_fonc(planes, out_planes, stride) 
        # else:
        #     
        self.conv = conv1x1_fonc(planes)

    def forward(self, x):
        y = self.conv(x)
        # if config_task.mode == 'series_adapters':
        #     y += x
        return y

class conv_task(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, nb_tasks=1, is_proj=1, second=0):
        super(conv_task, self).__init__()
        self.is_proj = is_proj
        self.second = second
        self.conv = conv3x3(in_planes, planes, stride)
        # if config_task.mode == 'series_adapters' and is_proj:
        #     self.bns = nn.ModuleList([nn.Sequential(conv1x1(planes), nn.BatchNorm2d(planes)) for i in range(nb_tasks)])
        # elif config_task.mode == 'parallel_adapters' and is_proj:
        #     self.parallel_conv = nn.ModuleList([conv1x1(in_planes, planes, stride) for i in range(nb_tasks)])
        #     self.bns = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])
        # else:
        #     self.bns = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])
        # self.bns = nn.ModuleList([film(planes) for i in range(nb_tasks)])
        self.bns = film(planes)
    
    # def forward(self, x):
    def forward(self, x, task=0):
        # task = config_task.task
        y = self.conv(x)
        # dropoutなし
        # if self.second == 0:
        #     if config_task.isdropout1:
        #         x = F.dropout2d(x, p=0.5, training = self.training)
        # else:
        #     if config_task.isdropout2:
        #         x = F.dropout2d(x, p=0.5, training = self.training)

        # if config_task.mode == 'parallel_adapters' and self.is_proj:
        #     y = y + self.parallel_conv[task](x)
        # y = self.bns[task](y)
        y = self.bns(y)

        return y

# No projection: identity shortcut
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=0, nb_tasks=1):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv_task(in_planes, planes, stride, nb_tasks, is_proj=int(config_task.proj[0]))
        # self.conv2 = nn.Sequential(nn.ReLU(True), conv_task(planes, planes, 1, nb_tasks, is_proj=int(config_task.proj[1]), second=1))
        self.conv1 = conv_task(in_planes, planes, stride, nb_tasks, is_proj=int('1'))
        self.conv2 = nn.Sequential(nn.ReLU(True), conv_task(planes, planes, 1, nb_tasks, is_proj=int('1'), second=1))
        self.shortcut = shortcut
        if self.shortcut == 1:
            self.avgpool = nn.AvgPool2d(2)
        
    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut == 1:
            residual = self.avgpool(x)
            residual = torch.cat((residual, residual*0),1)
        y += residual
        y = F.relu(y)
        return y


class ResNet(nn.Module):
    def __init__(self, block, nblocks, task_dict, fc, mode_norm='bn', version_film='film'):
        super(ResNet, self).__init__()
        nb_tasks = 1 # 使わない
        blocks = [block, block, block] # デフォルトではBasic blockが3つ
        # factor = config_task.factor
        factor = 1.0
        self.in_planes = int(32*factor)
        self.nc_list = [self.in_planes] + [int(64*factor)]*n_blocks[0]*2 + [int(128*factor)]*n_blocks[1]*2 + [int(256*factor)]*n_blocks[2]*2

        self.film_generator = film_generator(sum(self.nc_list), len(self.task_dict), fc)

        self.pre_layers_conv = conv_task(3,int(32*factor), 1, nb_tasks) 

        self.layer1 = self._make_layer(blocks[0], int(64*factor), nblocks[0], stride=2, nb_tasks=nb_tasks)
        self.layer2 = self._make_layer(blocks[1], int(128*factor), nblocks[1], stride=2, nb_tasks=nb_tasks)
        self.layer3 = self._make_layer(blocks[2], int(256*factor), nblocks[2], stride=2, nb_tasks=nb_tasks)

        self.end_bns = nn.Sequential(film(int(256*factor)),nn.ReLU(True))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.linears = nn.ModuleList([nn.Linear(int(256*factor), num_classes[i]) for i in range(nb_tasks)])   
        self.linear = nn.ModuleDict({})
        for task_name,item in self.task_dict.items():
            self.linear[task_name] = nn.Sequential(nn.Linear(int(256*factor), item['num_class']))      
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, nblocks, stride=1, nb_tasks=1):
        shortcut = 0
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = 1
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, nb_tasks=nb_tasks))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, nb_tasks=nb_tasks))
        return nn.Sequential(*layers)

    def forward(self, x, task=0, task_name=None, task_vec=None, visualize=False):
        x = self.pre_layers_conv(x)
        # task = config_task.task
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.end_bns[task](x)
        x = self.end_bns(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linears[task](x)
        return x

    def model_fit(self, x_pred, x_output, num_output, device):
        # # convert a single label into a one-hot vector
        # x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
        # x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        # # apply cross-entropy loss
        # loss = x_output_onehot * torch.log(x_pred + 1e-20)
        # loss = torch.sum(-loss, dim=1)
        # return torch.mean(loss)

        loss = nn.CrossEntropyLoss()(x_pred, x_output).to(device)
        return loss


def resnet26(num_classes=[10], blocks=BasicBlock):
    return  ResNet(blocks, [4,4,4],num_classes)

