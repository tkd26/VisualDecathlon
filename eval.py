import torch
import torch.nn as nn

import torch.nn.init as init
import torch.nn.functional as F
import torchvision
import numpy as np
import torch.optim as optim
import pickle
import argparse

from torchvision.transforms import transforms
from tqdm import tqdm

from sync_batchnorm import convert_model, DataParallelWithCallback
from model_WideRes import WideResNet as WideResNet
from model_ResNetbase import ResNetBaseNet

from utils.optimizer import *
from utils.data_transform import *

parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--random_lr', action='store_true')
parser.add_argument('--mode_model', default='WideRes', choices=['WideRes', 'ResNet18'])
parser.add_argument('--optim', default='adam', choices=['sgd', 'adam'])
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('--fc', type=int, default=5, choices=[1,3,5])
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--load', default=False, nargs='*')
parser.add_argument('--save_outputs', action='store_true')
parser.add_argument('--save_imgs_idx', type=int, default=None)
args = parser.parse_args()

def data_transform(data_path, name, train=True):
    with open(data_path + 'decathlon_mean_std.pickle', 'rb') as handle:
        dict_mean_std = pickle._Unpickler(handle)
        dict_mean_std.encoding = 'latin1'
        dict_mean_std = dict_mean_std.load()

    means = dict_mean_std[name + 'mean']
    stds = dict_mean_std[name + 'std']

    if name in ['gtsrb', 'omniglot', 'svhn']:  # no horz flip
        transform_train = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(72),
            transforms.RandomCrop(72),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if name in ['gtsrb', 'omniglot', 'svhn']:  # no horz flip
        transform_test = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if train:
        return transform_train
    else:
        return transform_test


# im_train_set = [0] * 10
# im_val_set = [0] * 10
im_test_set = [0] * 10
data_path = '/home/yanai-lab/takeda-m/space/dataset/decathlon-1.0/data/'
all_data_name = ['imagenet12', 'aircraft', 'cifar100', 'daimlerpedcls', 'dtd',
             'gtsrb', 'omniglot', 'svhn', 'ucf101', 'vgg-flowers']
# data_class = [1000, 100, 100, 2, 47, 43, 1623, 10, 101, 102]
data_name = ['aircraft', 'cifar100']
data_class = [100, 100]
task_num = len(data_name)
for i in range(10):
#     im_train_set[i] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(data_path + data_name[i] + '/train',
#                                                   transform=data_transform(data_path, data_name[i])),
#                                                   batch_size=128,
#                                                   shuffle=True,
#                                                   num_workers=4, pin_memory=True)
#     im_val_set[i] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(data_path + data_name[i] + '/val',
#                                                 transform=data_transform(data_path,data_name[i])),
#                                                 batch_size=128,
#                                                 shuffle=True,
#                                                 num_workers=4, pin_memory=True)
    im_test_set[i] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(data_path + all_data_name[i] + '/val',
                                                 transform=data_transform(data_path, all_data_name[i], train=False)),
                                                 batch_size=args.batch_size,
                                                 shuffle=False)
print('-----All dataset loaded-----')

# define WRN model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.mode_model=='WideRes':
    model = WideResNet(depth=28, widen_factor=4, num_classes=data_class, fc=args.fc).to(device)
elif args.mode_model=='ResNet18':
    model = ResNetBaseNet(data_class, args.fc).to(device)
    
# BatchNormalizeを含む場合のDataParallel
# model = convert_model(model).cuda()
# model = DataParallelWithCallback(model, device_ids=[0, 1, 2, 3]) 
# optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-5, nesterov=True, momentum=0.9)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# load parameter
if args.load:
    load_path, load_epoch = args.load
    load_epoch = int(load_epoch)
    param = torch.load('{}/{:0=5}.pth'.format(load_path, load_epoch))
    model.load_state_dict(param)
else:
    load_epoch = 0    

ans = {}
# task_vecs = { 
#     'imagenet12':   torch.Tensor([0,0,0,0,0,0,0,0,0]),
#     'aircraft':     torch.Tensor([1,0,0,0,0,0,0,0,0]),
#     'cifar100':      torch.Tensor([0,1,0,0,0,0,0,0,0]),
#     'daimlerpedcls':torch.Tensor([0,0,1,0,0,0,0,0,0]),
#     'dtd':          torch.Tensor([0,0,0,1,0,0,0,0,0]),
#     'gtsrb':        torch.Tensor([0,0,0,0,1,0,0,0,0]),
#     'omniglot':     torch.Tensor([0,0,0,0,0,1,0,0,0]),
#     'svhn':         torch.Tensor([0,0,0,0,0,0,1,0,0]),
#     'ucf101':       torch.Tensor([0,0,0,0,0,0,0,1,0]),
#     'vgg-flowers':  torch.Tensor([0,0,0,0,0,0,0,0,1]),
# }

task_vecs = {}
task_vecs_one_hot = torch.eye(task_num)
for i,task_name in enumerate(data_name):
    task_vecs[task_name] = task_vecs_one_hot[i]

model.eval()
with torch.no_grad():
    for k in range(0, 10):
        if all_data_name[k] in data_name:
            test_dataset = iter(im_test_set[k])
            test_batch = len(test_dataset)
            test_label = []
            for i in tqdm(range(test_batch)):
                test_data, _ = test_dataset.next()
                test_data = test_data.to(device)
                batch = test_data.shape[0]
                task_vec = task_vecs[all_data_name[k]].unsqueeze(0).repeat(batch,1).to(device)

                test_pred1 = model(test_data, data_name.index(all_data_name[k]), task_vec)

                # calculate testing loss and accuracy
                test_predict = test_pred1.data.max(1)[1]
                test_pred = test_predict.cpu().numpy()
                test_label.extend(test_pred)
            ans[all_data_name[k]] = test_label
            print(len(im_test_set[k]), len(ans[all_data_name[k]]))
        else:
#             data_num = len(im_test_set[k])
#             ans[all_data_name[k]] = [1]*data_num

            test_dataset = iter(im_test_set[k])
            test_batch = len(test_dataset)
            test_label = []
            for i in tqdm(range(test_batch)):
                test_data, _ = test_dataset.next()
                batch = test_data.shape[0]
                test_label.extend([0]*batch)
            ans[all_data_name[k]] = test_label

    #         print('DATASET: {:s} || TRAIN {:.4f} {:.4f} | TEST {:.4f} {:.4f}'
    #               .format(data_name[k], avg_cost[k][0], avg_cost[k][1], avg_cost[k][2], avg_cost[k][3]))
        print('Evaluating DATASET: {:s} ...'.format(all_data_name[k]))

pickle_out = open("ans.pickle", "wb")
pickle.dump(ans, pickle_out)
pickle_out.close()