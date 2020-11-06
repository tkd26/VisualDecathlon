import os, sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import math
    
class ConditionalNorm(nn.Module):
    def __init__(self, norm_nc, task_dict, mode_norm='bn'):
        super().__init__()

        if mode_norm=='bn':
            self.norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif mode_norm=='in':
            self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

        self.conv1x1 = {}
        for task in task_dict.keys():
            self.conv1x1[task] = nn.Conv2d(norm_nc, norm_nc, kernel_size=1, stride=1, padding=0).cuda()

    def forward(self, x, task_name=None):
        normalized = self.norm(x)
        out = self.conv1x1[task_name](normalized)
        return out
    

def conv3x3(in_planes, out_planes, stride=1):
    # res adaptersのresnet26ではbias=False
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1, task_dict=None, mode_norm='in', dropout=False):
        super(wide_basic, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
        self.norm1 = ConditionalNorm(in_planes, task_dict, mode_norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        
#         self.bn2 = nn.BatrchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.norm2 = ConditionalNorm(planes, task_dict, mode_norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

        self.dropout = dropout

    def forward(self, x, task_name):
        out1 = self.conv1(F.relu(self.norm1(x, task_name)))
        if self.dropout:
            out2 = self.conv2(F.dropout2d(F.relu(self.norm2(out1, task_name)), p=0.3))
        else:
            out2 = self.conv2(F.relu(self.norm2(out1, task_name)))
        out2 += self.shortcut(x.contiguous())
        return out2


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, task_dict, mode_norm='bn', dropout=False):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        self.n = int((depth - 4) / 6) # num blocks
        k = widen_factor
        filter = [16, 16 * k, 32 * k, 64 * k]
        self.task_dict = task_dict

        # input conv
        self.norm = ConditionalNorm(filter[0], task_dict, mode_norm)

        self.conv1 = conv3x3(3, filter[0], stride=1)
        
        # layer1
        depth = self.in_planes
        widen_factor = filter[1]
        stride = 2
        strides = [stride] + [1] * (self.n - 1)
        self.layer1 = nn.ModuleList([wide_basic(self.in_planes, filter[1], strides[0], self.task_dict, mode_norm, dropout)])
        for i in range(1, self.n):
            self.layer1.append(wide_basic(filter[1], filter[1], strides[i], self.task_dict, mode_norm, dropout))
        
        # layer2
        stride = 2
        strides = [stride] + [1] * (self.n - 1)
        self.layer2 = nn.ModuleList([wide_basic(filter[1], filter[2], strides[0], self.task_dict, mode_norm, dropout)])
        for i in range(1, self.n):
            self.layer2.append(wide_basic(filter[2], filter[2], strides[i], self.task_dict, mode_norm, dropout))
            
        # layer3
        stride = 2
        strides = [stride] + [1] * (self.n - 1)
        self.layer3 = nn.ModuleList([wide_basic(filter[2], filter[3], strides[0], self.task_dict, mode_norm, dropout)])
        for i in range(1, self.n):
            self.layer3.append(wide_basic(filter[3], filter[3], strides[i], self.task_dict, mode_norm, dropout))
        
#         self.bn1 = nn.BatchNorm2d(filter[3], momentum=0.9)
        self.norm_last = ConditionalNorm(filter[3], task_dict, mode_norm)
        
        # output layer
        self.linear = nn.ModuleDict({})
        for task,item in self.task_dict.items():
            self.linear[task] = nn.Sequential(
                nn.Linear(filter[3], item['num_class']),
                nn.Softmax(dim=1))

        # 重みとバイアスの初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine == True:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x, task_name, task_vec, visualize=False):
        
        g_encoder = [0] * 4
        # input conv
        param_idx = 0
        g_encoder[0] = self.norm(self.conv1(x), task_name)
        
        # layer1
        g_encoder[1] = self.layer1[0](g_encoder[0], task_name)

        g_encoder[1] = self.layer1[1](g_encoder[1], task_name)
        g_encoder[1] = self.layer1[2](g_encoder[1], task_name)
        g_encoder[1] = self.layer1[3](g_encoder[1], task_name)
                
        # layer2   
        g_encoder[2] = self.layer2[0](g_encoder[1], task_name)
        g_encoder[2] = self.layer2[1](g_encoder[2], task_name)
        g_encoder[2] = self.layer2[2](g_encoder[2], task_name)
        g_encoder[2] = self.layer2[3](g_encoder[2], task_name)
        
    
        # layer3   
        g_encoder[3] = self.layer3[0](g_encoder[2], task_name)
        g_encoder[3] = self.layer3[1](g_encoder[3], task_name)
        g_encoder[3] = self.layer3[2](g_encoder[3], task_name)
        g_encoder[3] = self.layer3[3](g_encoder[3], task_name)
                
        g_encoder[3] = F.relu(self.norm_last(g_encoder[3], task_name))

        pred = F.avg_pool2d(g_encoder[-1], 8)
        pred = pred.contiguous().view(pred.size(0), -1)

        out = self.linear[task_name](pred)
        return out

    def model_fit(self, x_pred, x_output, num_output, device):
        # convert a single label into a one-hot vector
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        # apply cross-entropy loss
        loss = x_output_onehot * torch.log(x_pred + 1e-20)
        loss = torch.sum(-loss, dim=1)
        return torch.mean(loss)