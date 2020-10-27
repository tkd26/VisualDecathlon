import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn

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
    def __init__(self, norm_nc, norm_mode='in'):
        super().__init__()
        
        '''
        ここでbatchかinstanceかを変える
        '''
        if norm_mode=='bn':
            self.norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif norm_mode=='in':
            self.norm = nn.InstanceNorm2d(norm_nc, affine=False)
            
    def forward(self, x, factor, bias):
        normalized = self.norm(x)
        out = normalized * factor + bias
        return out

class ResNetBaseNet(torch.nn.Module):
    def __init__(self, num_classes, fc, pretrained=True, ResNet='ResNet18', norm_mode='in'):
        super(ResNetBaseNet, self).__init__()
        self.ResNet = ResNet
        self.num_classes = num_classes
        self.task_num = len(num_classes)
        self.norm_mode = norm_mode
        
        # vgg model
        self.base_resnet, self.nc_list = self.make_ResNetBaseModel(pretrained=True)
#         print(self.nc_list)
        # film generator
        self.film_generator = film_generator(sum(self.nc_list), self.task_num, fc)
        
    #transfer model
    def make_ResNetBaseModel(self, pretrained):
        #load model
        if self.ResNet=='ResNet18':
            base_model = models.resnet18(pretrained=pretrained)
        blocks_ToPutFiLM = [base_model.bn1, base_model.layer1, 
                            base_model.layer2, base_model.layer3, base_model.layer4]
        
        nc_list = []
        for block in blocks_ToPutFiLM:
            if block==base_model.bn1: 
                cond_nc = 64
                base_model.bn1 = film(cond_nc, self.norm_mode)
                nc_list += [cond_nc]
            else:
                cond_nc = block[0].conv1.out_channels
                block[0].bn1 = film(cond_nc, self.norm_mode)
                block[0].bn2 = film(cond_nc, self.norm_mode)
                block[1].bn1 = film(cond_nc, self.norm_mode)
                block[1].bn2 = film(cond_nc, self.norm_mode)
                nc_list += [cond_nc] * 4
                if 'downsample' in str(list(block[0].modules())[0]):
                    block[0].downsample[1] = film(cond_nc, self.norm_mode)
                    nc_list += [cond_nc]
        base_model.avgpool = nn.AdaptiveAvgPool2d((2,2))
        
        fc_nc = 512*2*2
        base_model.fc = nn.ModuleList([nn.Sequential(
                nn.Linear(fc_nc, self.num_classes[0]),
                nn.Softmax(dim=1))])
        for i in range(1, self.task_num):
            base_model.fc.append(nn.Sequential(nn.Linear(fc_nc, self.num_classes[i]),
                                                     nn.Softmax(dim=1)))
#         print(base_model)
        return base_model, nc_list

    def _forward_layers(self, x, factor_list, bias_list):
        base_model = self.base_resnet
        blocks = [base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool,
                            base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4]
        i = 0
        for block in blocks:
            if block==base_model.bn1:
                x = block(x, factor_list[i], bias_list[i])
                i += 1
            elif block in [base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4] :
                x_id = x
                for j in range(2):
                    basic_block = block[j]
                    if 'downsample' in str(list(basic_block.modules())[0]):
                        layer_blocks = [basic_block.conv1, basic_block.bn1, basic_block.relu, 
                                        basic_block.conv2, basic_block.bn2, 
                                        basic_block.downsample]
                    else:
                        layer_blocks = [basic_block.conv1, basic_block.bn1, basic_block.relu, 
                                        basic_block.conv2, basic_block.bn2]
                        
                    for layer_block in layer_blocks:
                        if len(layer_blocks)==5:
                            if layer_block in [basic_block.bn1, basic_block.bn2]:
                                x = layer_block(x, factor_list[i], bias_list[i])
                                i += 1
                            else:
                                x = layer_block(x)
                                if layer_block==layer_blocks[-1]:
                                    x += x_id
                        elif len(layer_blocks)==6:
                            if layer_block==basic_block.downsample:
                                x_id = basic_block.downsample[0](x_id)
                                x_id = basic_block.downsample[1](x_id, factor_list[i], bias_list[i])
                                i += 1
                            elif layer_block in [basic_block.bn1, basic_block.bn2]:
                                x = layer_block(x, factor_list[i], bias_list[i])
                                i += 1
                            else:
                                x = layer_block(x)
                x += x_id
            else:
                x = block(x)

        return x

    def forward(self, x, k, cond):
        # film generator
        factor, bias = self.film_generator(cond)
        factor_list, bias_list = [],[]
        s = 0
        for nc in self.nc_list:
            factor_list.append(factor[:,s:s+nc,:,:])
            bias_list.append(bias[:,s:s+nc,:,:])
            s += nc
        
        x = self._forward_layers(x, factor_list, bias_list)
#         print('-----{}-----'.format(x.shape))
        x = self.base_resnet.avgpool(x)
        x = x.contiguous().view(x.size(0), -1)

        x = self.base_resnet.fc[k](x)
        return x
    
    def model_fit(self, x_pred, x_output, num_output, device):
        # convert a single label into a one-hot vector
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        # apply cross-entropy loss
        loss = x_output_onehot * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)
   
# model = ResNetBaseNet(num_classes=[1000, 100, 100, 2, 47, 43, 1623, 10, 101, 102])
# data = torch.randn(5,3,72,72)
# cond = torch.Tensor([0,0,0,0,0,0,0,0,0]).unsqueeze(0).repeat(5,1)
# x = model(data, 0, cond)
# print(x.shape)