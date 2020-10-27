# -*- coding: utf-8 -*-
import os, sys
import argparse

# proxy
os.environ["http_proxy"] = "http://proxy.uec.ac.jp:8080/"
os.environ["https_proxy"] = "http://proxy.uec.ac.jp:8080/"

import torch
import torch.nn as nn

import torch.nn.init as init
import torch.nn.functional as F
import torchvision
import numpy as np
import torch.optim as optim
import pickle
import tensorboardX as tbx
from tqdm import tqdm

from torchvision.transforms import transforms

from sync_batchnorm import convert_model, DataParallelWithCallback
from models.model_WideRes import WideResNet as WideResNet
from models.model_WideRes_mask import WideResNet as WideResNet_mask
from models.model_WideRes_STL import WideResNet as WideResNet_STL
from models.model_ResNetbase import ResNetBaseNet

from utils.optimizer import *
from utils.data_transform import *

parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--random_lr', action='store_true') # 学習率をチャネルごとに設定するか
parser.add_argument('--mode', default='train', choices=['train', 'val'])
parser.add_argument('--mode_model', default='WideRes', choices=['WideRes','WideRes_mask', 'WideRes_STL', 'ResNet18'])
parser.add_argument('--optim', default='adam', choices=['sgd', 'sgd2', 'sgd3', 'sgd3-2', 'sgd3-3', 'sgd3-4', 'sgd3-5', 'adam'])
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('--fc', type=int, default=5, choices=[1,3,5])
parser.add_argument('--norm', type=str, default='bn', choices=['in', 'bn'])
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--version', type=str, default=None)
parser.add_argument('--load', default=False, nargs='*')
# parser.add_argument('--save_outputs', action='store_true')
# parser.add_argument('--save_imgs_idx', type=int, default=None)

args = parser.parse_args()

if args.gpu==None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ['0', '1', '2', '3']
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

'''
------------------------------------------------------------------------
データの初期設定
------------------------------------------------------------------------
'''
data_path = '/home/yanai-lab/takeda-m/space/dataset/decathlon-1.0/data/'
data_names = ['imagenet12', 'aircraft', 'cifar100', 'daimlerpedcls', 'dtd',
             'gtsrb', 'omniglot', 'svhn', 'ucf101', 'vgg-flowers']
data_classes = [1000, 100, 100, 2, 47, 43, 1623, 10, 101, 102]


if args.version in data_names:
    data_name = [args.version]
    data_class = [data_classes[data_names.index(args.version)]]
elif args.version == 'cifar100_ucf101':
    data_name = ['cifar100', 'ucf101']
    data_class = [100, 101]  
elif args.version == '5tasks':
    data_name = ['aircraft', 'cifar100', 'daimlerpedcls', 'dtd', 'gtsrb']
    data_class = [100, 100, 2, 47, 43]
    
task_num = len(data_name)

'''
------------------------------------------------------------------------
保存ファイルの設定
------------------------------------------------------------------------
'''
# save name
save_name = []
if args.random_lr: save_name.extend(['RandomLR'])
save_name.extend(['OPTIM'+args.optim])
save_name.extend(['Model'+args.mode_model])
if args.fc!=None: save_name.extend(['FC{}'.format(args.fc)])
if args.batch_size!=128: save_name.extend(['BS{}'.format(args.batch_size)])
# save_name.extend([args.norm])
if args.version: save_name.extend([args.version])
save_name.extend(data_name)
save_name = '_'.join(save_name)

# make folder to save network
if not os.path.exists('./weights/{}'.format(save_name)):
    os.makedirs('./weights/{}'.format(save_name))

'''
------------------------------------------------------------------------
データロード
------------------------------------------------------------------------
'''
im_train_set = [0] * task_num
im_test_set = [0] * task_num
for i in range(task_num):
    im_train_set[i] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(data_path + data_name[i] + '/train',
                                                  transform=data_transform(data_path,data_name[i])),
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=4, pin_memory=True)
    im_test_set[i] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(data_path + data_name[i] + '/val',
                                                 transform=data_transform(data_path,data_name[i], train=False)),
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=4, pin_memory=True)
    print('{} loaded'.format(data_name[i]))
print('-----All dataset loaded-----')


'''
------------------------------------------------------------------------
モデルの設定
------------------------------------------------------------------------
'''
# define WRN model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.mode_model=='WideRes':
    model = WideResNet(depth=28, widen_factor=4, num_classes=data_class, fc=args.fc, mode_norm=args.norm).to(device)
elif args.mode_model=='WideRes_mask':
    model = WideResNet_mask(depth=28, widen_factor=4, num_classes=data_class, fc=args.fc, mode_norm=args.norm).to(device)
elif args.mode_model=='WideRes_STL':
    model = WideResNet_STL(depth=28, widen_factor=4, num_classes=data_class, fc=args.fc).to(device)
elif args.mode_model=='ResNet18':
    model = ResNetBaseNet(data_class, args.fc).to(device)


# random_list =  [1e-3, 1e-4, 1e-5]
random_list =  [0, 1] # version 3
if args.random_lr:
    if args.optim=='sgd':
        optimizer = SGD_c(params=[ # lambda:共有率
                {"params": model.film_generator.parameters(), "lambda": 1.0},
                {"params": model.film.parameters(), "lambda": 1.0},
                {"params": model.conv1.parameters(), "lambda": 0.0},
                {"params": model.layer1.parameters(), "lambda": 0.0},
                {"params": model.layer2.parameters(), "lambda": 0.0},
                {"params": model.layer3.parameters(), "lambda": 0.0},
                {"params": model.film_last.parameters(), "lambda": 1.0},
                {"params": model.linear.parameters(), "lambda": 1.0},
        ], do_task_list=data_name)
    elif args.optim=='sgd2':
            optimizer = SGD_c(params=[
                {"params": model.film_generator.parameters(), "lambda": 1.0},
                {"params": model.film.parameters(), "lambda": 1.0},
                {"params": model.conv1.parameters(), "lambda": 0.0},
                {"params": model.layer1.parameters(), "lambda": 0.0},
                {"params": model.layer2.parameters(), "lambda": 0.0},
                {"params": model.layer3.parameters(), "lambda": 0.0},
                {"params": model.film_last.parameters(), "lambda": 1.0},
                {"params": model.linear.parameters(), "lambda": 1.0},],
                              lr=0.1, weight_decay=5e-5, nesterov=True, momentum=0.9, 
                              do_task_list=data_name)
    elif args.optim=='sgd3':
            optimizer = SGD_c(params=[
                {"params": model.film_generator.parameters(), "lambda": 1.0},
                {"params": model.film.parameters(), "lambda": 1.0},
                {"params": model.conv1.parameters(), "lambda": 0.0},
                {"params": model.layer1.parameters(), "lambda": 0.0},
                {"params": model.layer2.parameters(), "lambda": 0.0},
                {"params": model.layer3.parameters(), "lambda": 0.0},
                {"params": model.film_last.parameters(), "lambda": 1.0},
                {"params": model.linear.parameters(), "lambda": 1.0},],
                              lr=0.01, weight_decay=0, nesterov=True, momentum=0.5,
                              do_task_list=data_name)
    elif args.optim=='sgd3-2':
            optimizer = SGD_c(params=[
                {"params": model.film_generator.parameters(), "lambda": 1.0},
                {"params": model.film.parameters(), "lambda": 1.0},
                {"params": model.conv1.parameters(), "lambda": 1.0},
                {"params": model.layer1.parameters(), "lambda": 1.0},
                {"params": model.layer2.parameters(), "lambda": 0.5},
                {"params": model.layer3.parameters(), "lambda": 0.0},
                {"params": model.film_last.parameters(), "lambda": 1.0},
                {"params": model.linear.parameters(), "lambda": 1.0},],
                              lr=0.01, weight_decay=0, nesterov=True, momentum=0.5,
                              do_task_list=data_name)
    elif args.optim=='sgd3-3':
            optimizer = SGD_c(params=[
                {"params": model.film_generator.parameters(), "lambda": 1.0},
                {"params": model.film.parameters(), "lambda": 1.0},
                {"params": model.conv1.parameters(), "lambda": 1.0},
                {"params": model.layer1.parameters(), "lambda": 1.0},
                {"params": model.layer2.parameters(), "lambda": 2/3},
                {"params": model.layer3.parameters(), "lambda": 1/3},
                {"params": model.film_last.parameters(), "lambda": 1.0},
                {"params": model.linear.parameters(), "lambda": 1.0},],
                              lr=0.01, weight_decay=0, nesterov=True, momentum=0.5,
                              do_task_list=data_name)
    elif args.optim=='sgd3-4':
            optimizer = SGD_c(params=[
                {"params": model.film_generator.parameters(), "lambda": 1.0},
                {"params": model.film.parameters(), "lambda": 1.0},
                {"params": model.conv1.parameters(), "lambda": 1.0},
                {"params": model.layer1.parameters(), "lambda": 1.0},
                {"params": model.layer2.parameters(), "lambda": 1.0},
                {"params": model.layer3.parameters(), "lambda": 0.5},
                {"params": model.film_last.parameters(), "lambda": 1.0},
                {"params": model.linear.parameters(), "lambda": 1.0},],
                              lr=0.01, weight_decay=0, nesterov=True, momentum=0.5,
                              do_task_list=data_name)
    elif args.optim=='sgd3-5':
            optimizer = SGD_c(params=[
                {"params": model.film_generator.parameters(), "lambda": 1.0},
                {"params": model.film.parameters(), "lambda": 1.0},
                {"params": model.conv1.parameters(), "lambda": 1.0},
                {"params": model.layer1.parameters(), "lambda": 1.0},
                {"params": model.layer2.parameters(), "lambda": 1.0},
                {"params": model.layer3.parameters(), "lambda": 0.0},
                {"params": model.film_last.parameters(), "lambda": 1.0},
                {"params": model.linear.parameters(), "lambda": 1.0},],
                              lr=0.01, weight_decay=0, nesterov=True, momentum=0.5,
                              do_task_list=data_name)
    else:
        print('optimizer not selected.')
        sys.exit()
            
else:
    if args.optim=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=0, momentum=0.9)
    elif args.optim=='sgd2':
        optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-5, nesterov=True, momentum=0.9)
    elif args.optim=='sgd3':
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0, nesterov=True, momentum=0.5)
    elif args.optim=='adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    else:
        print('optimizer not selected.')
        sys.exit()

# BatchNormalizeを含む場合のDataParallel
# model = model.module.cpu()
# model = convert_model(model).cuda()
# model = DataParallelWithCallback(model, device_ids=[0, 1, 2, 3])
# model = torch.nn.DataParallel(model)

# load parameter
if args.load:
    load_path, load_epoch = args.load
    load_epoch = int(load_epoch)
    param = torch.load('{}/{:0=5}.pth'.format(load_path, load_epoch))
    model.load_state_dict(param)
else:
    load_epoch = 0
    
# define writer
if args.mode=='train':
    writer = tbx.SummaryWriter(log_dir='logs/{}'.format(save_name))

# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Parameter Space: ABS: {:.1f}'.format(count_parameters(model)))

'''
------------------------------------------------------------------------
学習の設定
------------------------------------------------------------------------
'''
# define parameters
total_epoch = 1000
first_run = True
avg_cost = np.zeros([total_epoch, task_num, 4], dtype=np.float32)
# task_vecs = { 
#     'imagenet12':   torch.Tensor([1,0,0,0,0,0,0,0,0,0]),
#     'aircraft':     torch.Tensor([0,1,0,0,0,0,0,0,0,0]),
#     'cifar100':      torch.Tensor([0,0,1,0,0,0,0,0,0,0]),
#     'daimlerpedcls':torch.Tensor([0,0,0,1,0,0,0,0,0,0]),
#     'dtd':          torch.Tensor([0,0,0,0,1,0,0,0,0,0]),
#     'gtsrb':        torch.Tensor([0,0,0,0,0,1,0,0,0,0]),
#     'omniglot':     torch.Tensor([0,0,0,0,0,0,1,0,0,0]),
#     'svhn':         torch.Tensor([0,0,0,0,0,0,0,1,0,0]),
#     'ucf101':       torch.Tensor([0,0,0,0,0,0,0,0,1,0]),
#     'vgg-flowers':  torch.Tensor([0,0,0,0,0,0,0,0,0,1]),
# }
# task vectors
task_vecs = {}
task_vecs_one_hot = torch.eye(task_num)
for i,task_name in enumerate(data_name):
    task_vecs[task_name] = task_vecs_one_hot[i]
print('TASK VECS:', task_vecs)
    

'''
------------------------------------------------------------------------
学習
------------------------------------------------------------------------
'''
max_acc = [0] * task_num
max_acc_epoch = [0] * task_num
max_avg_acc = [0] * task_num
max_avg_acc_epoch = 0

# running
for index in range(load_epoch, total_epoch):

    for k in range(task_num):
        
        cost = np.zeros(2, dtype=np.float32)
        train_dataset = iter(im_train_set[k])
        train_batch = len(train_dataset)

        if args.mode == 'train':
            print('train')
            model.train()
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                
                for i in range(train_batch):
                    
                    train_data, train_label = train_dataset.next()
                    train_label = train_label.type(torch.LongTensor)
                    train_data, train_label = train_data.to(device), train_label.to(device)

                    batch = train_data.shape[0]
                    task_vec = task_vecs[data_name[k]].unsqueeze(0).repeat(batch,1).to(device)

                    train_pred1 = model(train_data, k, task_vec)

                    # reset optimizer with zero gradient
                    
                    train_loss1 = model.model_fit(train_pred1, train_label, num_output=data_class[k], device=device)
                    train_loss = torch.mean(train_loss1)
                    train_loss.backward()
                    if args.random_lr:
                        optimizer.step(task_idx=k)
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    # calculate training loss and accuracy
                    train_predict_label1 = train_pred1.data.max(1)[1]
                    train_acc1 = train_predict_label1.eq(train_label).sum().item() / train_data.shape[0]

                    cost[0] = torch.mean(train_loss1).item()
                    cost[1] = train_acc1
                    avg_cost[index][k][0:2] += cost / train_batch
                    
        if args.mode == 'train' or args.mode == 'val':
            # evaluating test data
            print('val')
            with torch.no_grad():
                model.eval()
                test_dataset = iter(im_test_set[k])
                test_batch = len(test_dataset)
                for i in range(test_batch):
                    test_data, test_label = test_dataset.next()
                    test_label = test_label.type(torch.LongTensor)
                    test_data, test_label = test_data.to(device), test_label.to(device)

                    batch = test_data.shape[0]
                    task_vec = task_vecs[data_name[k]].unsqueeze(0).repeat(batch,1).to(device)

                    test_pred1 = model(test_data, k, task_vec)

                    test_loss1 = model.model_fit(test_pred1, test_label, num_output=data_class[k], device=device)
                    test_loss = torch.mean(test_loss1)

                    # calculate testing loss and accuracy
                    test_predict_label1 = test_pred1.data.max(1)[1]
                    test_acc1 = test_predict_label1.eq(test_label).sum().item() / test_data.shape[0]

                    cost[0] = torch.mean(test_loss1).item()
                    cost[1] = test_acc1
                    avg_cost[index][k][2:] += cost / test_batch
        
        if avg_cost[index][k][3] > max_acc[k]:
            max_acc[k] = avg_cost[index][k][3]
            max_acc_epoch[k] = index + 1
        print('EPOCH: {:04d} | DATASET: {:s} || TRAIN: {:.4f} {:.4f} || TEST: {:.4f} {:.4f} || MAX: {:.4f} ({:04d}ep)'
              .format(index+1, data_name[k],
                      avg_cost[index][k][0], avg_cost[index][k][1],
                      avg_cost[index][k][2], avg_cost[index][k][3],
                      max_acc[k], max_acc_epoch[k]))
        
    if sum(avg_cost[index, :, 3]) > sum(max_avg_acc[:]):
        max_avg_acc[:] = avg_cost[index, :, 3]
        max_avg_acc_epoch = index + 1
    avg_acc_txt = '({:04d}ep)'.format(max_avg_acc_epoch)
    for i in range(task_num):
        avg_acc_txt += '{}: {:.4f}, '.format(data_name[i], max_avg_acc[i])
    print(avg_acc_txt)
    
    print('===================================================')
    torch.save(model.state_dict(), 'weights/{}/{:0=5}.pth'.format(save_name, index+1))
        
    if args.mode=='val': 
        break
    else:     
        # write tensorboardX
        train_loss = dict([(data_name[i], avg_cost[index][i][0]) for i in range(task_num)])
        train_acc = dict([(data_name[i], avg_cost[index][i][1]) for i in range(task_num)])
        val_loss = dict([(data_name[i], avg_cost[index][i][2]) for i in range(task_num)])
        val_acc = dict([(data_name[i], avg_cost[index][i][3]) for i in range(task_num)])
        writer.add_scalars('train_loss', train_loss, index+1)
        writer.add_scalars('train_acc', train_acc, index+1)
        writer.add_scalars('val_loss', val_loss, index+1)
        writer.add_scalars('val_acc', val_acc, index+1)
        
if args.mode=='train': writer.close()

