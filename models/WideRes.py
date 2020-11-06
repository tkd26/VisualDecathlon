import os, sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import math

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
            if visualize:
                view_factor = torch.squeeze(factor[0])
                view_bias = torch.squeeze(bias[0])
                print(k, factor.shape, bias.shape)
                # 画像のプロット先の準備
                fig = plt.figure()
                plt.bar(range(1, len(view_factor)+1), view_factor.cpu().numpy(), ec='black')
                plt.ylim(0, 4)
                plt.title("normal histogram")
                fig.savefig("/home/yanai-lab/takeda-m/space/jupyter/notebook/Multi-Task-Learning/VisualDecathlon_lr/graph/{}img.png".format(k))
                fig2 = plt.figure()
                plt.bar(range(1, len(view_bias)+1), view_bias.cpu().numpy(), ec='black')
                plt.ylim(-3, 3)
                plt.title("normal histogram")
                fig2.savefig("/home/yanai-lab/takeda-m/space/jupyter/notebook/Multi-Task-Learning/VisualDecathlon_lr/graph/{}img_bias.png".format(k))
        elif self.version_film=='binary':
            factor_binary = torch.where(factor > 0, torch.ones(1).cuda(), torch.zeros(1).cuda())
            out = normalized * factor + bias 
        elif self.version_film=='no_film':
            out = normalized
        return out
    

def conv3x3(in_planes, out_planes, stride=1):
    # res adaptersのresnet26ではbias=False
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1, mode_norm='in', version_film='film', dropout):
        super(wide_basic, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
        self.film1 = film(in_planes, mode_norm, version_film)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        
#         self.bn2 = nn.BatrchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.film2 = film(planes, mode_norm, version_film)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

        self.dropout = dropout

    def forward(self, x, factor=None, bias=None, visualize=False, k=None):
        out1 = self.conv1(F.relu(self.film1(x, factor[0], bias[0], visualize=visualize, k=k)))
        if self.dropout:
            out2 = self.conv2(F.dropout2d(F.relu(self.film2(out1, factor[1], bias[1])), p=0.3))
        else:
            out2 = self.conv2(F.relu(self.film2(out1, factor[1], bias[1])))
        out2 += self.shortcut(x.contiguous())
        return out2


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, task_dict, fc, mode_norm='bn', version_film='film', dropout=False):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        self.n = int((depth - 4) / 6) # num blocks
        k = widen_factor
        filter = [16, 16 * k, 32 * k, 64 * k]
        self.task_dict = task_dict
        self.version_film = version_film
        
        self.nc_list = [filter[0], # init conv
                        filter[0],filter[1],filter[1],filter[1],filter[1],filter[1],filter[1],filter[1], # layer1
                        filter[1],filter[2],filter[2],filter[2],filter[2],filter[2],filter[2],filter[2], # layer2
                        filter[2],filter[3],filter[3],filter[3],filter[3],filter[3],filter[3],filter[3], # layer3
                        filter[3]] # last conv
        self.film_generator = film_generator(sum(self.nc_list), len(self.task_dict), fc)
        # input conv
        self.film = film(filter[0], mode_norm, self.version_film)

        self.conv1 = conv3x3(3, filter[0], stride=1)
        
        # layer1
        depth = self.in_planes
        widen_factor = filter[1]
        stride = 2
        strides = [stride] + [1] * (self.n - 1)
        self.layer1 = nn.ModuleList([wide_basic(self.in_planes, filter[1], strides[0], mode_norm, self.version_film, dropout)])
        for i in range(1, self.n):
            self.layer1.append(wide_basic(filter[1], filter[1], strides[i], mode_norm, self.version_film, dropout))
        
        # layer2
        stride = 2
        strides = [stride] + [1] * (self.n - 1)
        self.layer2 = nn.ModuleList([wide_basic(filter[1], filter[2], strides[0], mode_norm, self.version_film, dropout)])
        for i in range(1, self.n):
            self.layer2.append(wide_basic(filter[2], filter[2], strides[i], mode_norm, self.version_film, dropout))
            
        # layer3
        stride = 2
        strides = [stride] + [1] * (self.n - 1)
        self.layer3 = nn.ModuleList([wide_basic(filter[2], filter[3], strides[0], mode_norm, self.version_film, dropout)])
        for i in range(1, self.n):
            self.layer3.append(wide_basic(filter[3], filter[3], strides[i], mode_norm, self.version_film, dropout))
        
#         self.bn1 = nn.BatchNorm2d(filter[3], momentum=0.9)
        self.film_last = film(filter[3], mode_norm, self.version_film)
        
        # output layer
        self.linear = nn.ModuleDict({})
        for task_name,item in self.task_dict.items():
            self.linear[task_name] = nn.Sequential(
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
        # film generator
        factor, bias = self.film_generator(task_vec)
        factor_list, bias_list = [],[]
        s = 0
        for nc in self.nc_list:
            factor_list.append(factor[:,s:s+nc,:,:])
            bias_list.append(bias[:,s:s+nc,:,:])
            s += nc
        
        g_encoder = [0] * 4
        # input conv
        param_idx = 0
        g_encoder[0] = self.film(self.conv1(x), factor_list[0], bias_list[0])
        
        # layer1
        g_encoder[1] = self.layer1[0](g_encoder[0], factor_list[1:3], bias_list[1:3])

        g_encoder[1] = self.layer1[1](g_encoder[1], factor_list[3:5], bias_list[3:5])
        g_encoder[1] = self.layer1[2](g_encoder[1], factor_list[5:7], bias_list[5:7])
        g_encoder[1] = self.layer1[3](g_encoder[1], factor_list[7:9], bias_list[7:9])
                
        # layer2   
        g_encoder[2] = self.layer2[0](g_encoder[1], factor_list[9:11], bias_list[9:11])
        g_encoder[2] = self.layer2[1](g_encoder[2], factor_list[11:13], bias_list[11:13],  visualize=visualize, k=task_name)
        g_encoder[2] = self.layer2[2](g_encoder[2], factor_list[13:15], bias_list[13:15])
        g_encoder[2] = self.layer2[3](g_encoder[2], factor_list[15:17], bias_list[15:17])
        
    
        # layer3   
        g_encoder[3] = self.layer3[0](g_encoder[2], factor_list[17:19], bias_list[17:19])
        g_encoder[3] = self.layer3[1](g_encoder[3], factor_list[19:21], bias_list[19:21])
        g_encoder[3] = self.layer3[2](g_encoder[3], factor_list[21:23], bias_list[21:23])
        g_encoder[3] = self.layer3[3](g_encoder[3], factor_list[23:25], bias_list[23:25])
                
        g_encoder[3] = F.relu(self.film_last(g_encoder[3], factor_list[25], bias_list[25]))

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