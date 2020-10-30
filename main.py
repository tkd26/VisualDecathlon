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

parser.add_argument('--imagenet12', action='store_true')
parser.add_argument('--aircraft', action='store_true')
parser.add_argument('--cifar100', action='store_true')
parser.add_argument('--daimlerpedcls', action='store_true')
parser.add_argument('--dtd', action='store_true')
parser.add_argument('--gtsrb', action='store_true')
parser.add_argument('--omniglot', action='store_true')
parser.add_argument('--svhn', action='store_true')
parser.add_argument('--ucf101', action='store_true')
parser.add_argument('--vgg_flowers', action='store_true')

parser.add_argument('--random_lr', action='store_true') # 学習率をチャネルごとに設定するか
parser.add_argument('--mode', default='train', choices=['train', 'val', 'test'])
parser.add_argument('--mode_model', default='WideRes', choices=['WideRes','WideRes_mask', 'WideRes_STL', 'WideRes_pretrain', 'ResNet18'])
parser.add_argument('--optim', default='adam', choices=['sgd', 'sgd2', 'sgd3', 'sgd3-2', 'sgd3-3', 'sgd3-4', 'sgd3-5', 'sgd_pre', 'sgd_pre2', 'adam'])
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('--fc', type=int, default=5, choices=[1,3,5])
parser.add_argument('--norm', type=str, default='bn', choices=['in', 'bn'])
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--version', type=str, default=None)
parser.add_argument('--load', default=False, nargs='*')
parser.add_argument('--visualize', action='store_true') # filmパラメータを可視化
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

'''
    task_dict：10タスクすべての情報格納した辞書データ
        do：そのタスクを学習するか
        num_class：タスクの出力クラス数
        task_vec：タスク指定ベクトル
    task_num：全タスク数（=10）．task_dictの長さ
    do_task_list：実際に学習するタスクのリスト．train/valでは指定タスク，testでは全タスクが選択される．
'''

data_path = '/home/yanai-lab/takeda-m/space/dataset/decathlon-1.0/data/'

task_dict = {
    'imagenet12': {
        'do': args.imagenet12,
        'num_class': 1000,
        'task_vec': torch.Tensor([1,0,0,0,0,0,0,0,0,0])},
    'aircraft': {
        'do': args.aircraft,
        'num_class': 100,
        'task_vec': torch.Tensor([0,1,0,0,0,0,0,0,0,0])},
    'cifar100': {
        'do': args.cifar100,
        'num_class': 100,
        'task_vec': torch.Tensor([0,0,1,0,0,0,0,0,0,0])},
    'daimlerpedcls': {
        'do': args.daimlerpedcls,
        'num_class': 2,
        'task_vec': torch.Tensor([0,0,0,1,0,0,0,0,0,0])},
    'dtd': {
        'do': args.dtd,
        'num_class': 47,
        'task_vec': torch.Tensor([0,0,0,0,1,0,0,0,0,0])},
    'gtsrb': {
        'do': args.gtsrb,
        'num_class': 43,
        'task_vec': torch.Tensor([0,0,0,0,0,1,0,0,0,0])},
    'omniglot': {
        'do': args.omniglot,
        'num_class': 1623,
        'task_vec': torch.Tensor([0,0,0,0,0,0,1,0,0,0])},
    'svhn': {
        'do': args.svhn,
        'num_class': 10,
        'task_vec': torch.Tensor([0,0,0,0,0,0,0,1,0,0])},
    'ucf101': {
        'do': args.ucf101,
        'num_class': 101,
        'task_vec': torch.Tensor([0,0,0,0,0,0,0,0,1,0])},
    'vgg-flowers': {
        'do': args.vgg_flowers,
        'num_class': 102,
        'task_vec': torch.Tensor([0,0,0,0,0,0,0,0,0,1])},
}

if args.mode == 'train' or args.mode == 'val': # train時は選択したタスクのみを学習
    do_task_list = [name for name,item in task_dict.items() if item['do']==True]
elif args.mode == 'test': # test時はすべてのタスクを評価
    do_task_list = [name for name,item in task_dict.items()]
    
task_num = len(task_dict)

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
# save_name.extend(data_name)
save_name.extend(do_task_list)
save_name = '_'.join(save_name)

# make folder to save network
if args.mode == 'train':
    if not os.path.exists('./weights/{}'.format(save_name)):
        os.makedirs('./weights/{}'.format(save_name))

'''
------------------------------------------------------------------------
データロード
------------------------------------------------------------------------
'''
'''
    データセットは，do_task_listに記載のあるタスクのデータのみロードする．
    im_train_set：学習データ
    im_test_set：testと名前がついているが，中身はvalデータ（実際のtestデータは教師ラベルが与えられてないから）
'''
im_train_set = {}
im_test_set = {}
for task_name in do_task_list:
    im_train_set[task_name] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(data_path + task_name + '/train',
                                                  transform=data_transform(data_path,task_name)),
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=4, pin_memory=True)
    im_test_set[task_name] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(data_path + task_name + '/val',
                                                 transform=data_transform(data_path,task_name, train=False)),
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=4, pin_memory=True)
    print('{} loaded'.format(task_name))
print('-----All dataset loaded-----')


'''
------------------------------------------------------------------------
モデルの設定
------------------------------------------------------------------------
'''
'''
    modelは基本的にWideResを使用
        WideRes：ベーシックモデル．film+lr学習時に使用．
        WideRes_mask：film（バイナリマスク）+lr学習時に使用
        WideRes_STL：シングルタスク学習時に使用
        WideRes_pretrain：プレトレインモデルによるシングルタスク学習時に使用
                        （出力レイヤがプレトレインモデルと異なり，学習タスクに対応するようになっている．filmはなし）
                        
            task_dict：モデルに学習させるタスク情報．
                        追加ですべてのタスクを学習可能にするために，task_dict（10タスク分の情報）をそのまま入れる．

    DataParallelは，BN使用時は基本的にしない
    train時のみ，tensorboardに学習結果を書き込む
'''
# define WRN model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.mode_model=='WideRes':
    model = WideResNet(depth=28, widen_factor=4, task_dict=task_dict, fc=args.fc, mode_norm=args.norm, version_film='film').to(device)
elif args.mode_model=='WideRes_mask':
    # model = WideResNet_mask(depth=28, widen_factor=4, num_classes=data_class, fc=args.fc, mode_norm=args.norm).to(device)
    model = WideResNet(depth=28, widen_factor=4, num_classes=data_class, fc=args.fc, mode_norm=args.norm, version_film='binary').to(device)
elif args.mode_model=='WideRes_STL':
    model = WideResNet_STL(depth=28, widen_factor=4, num_classes=data_class, fc=args.fc).to(device)
elif args.mode_model=='WideRes_pretrain':
    model = WideResNet(depth=28, widen_factor=4, task_dict=task_dict, fc=args.fc, mode_norm=args.norm, with_film=False, version_film='no_film').to(device)
elif args.mode_model=='ResNet18':
    model = ResNetBaseNet(data_class, args.fc).to(device)

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
        ], do_task_list=do_task_list)
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
                              do_task_list=do_task_list)
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
                              do_task_list=do_task_list)
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
                              do_task_list=do_task_list)
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
                              do_task_list=do_task_list)
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
                              do_task_list=do_task_list)
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
                              do_task_list=do_task_list)
    elif args.optim=='sgd_pre':
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
                              do_task_list=['imagenet12', 'aircraft', 'cifar100', 'daimlerpedcls', 'dtd', 'gtsrb', 'omniglot', 'svhn', 'ucf101', 'vgg-flowers'])
    elif args.optim=='sgd_pre2': # layer3（lambda=0.0にしたところ）だけimagenetで0.5学習される
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
                              do_task_list=['imagenet12', 'dummy'])
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
if args.mode =='train':
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

# avg_cost = np.zeros([total_epoch, task_num, 4], dtype=np.float32)

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
# task_vecs = {}
# task_vecs_one_hot = torch.eye(task_num)
# for i,task_name in enumerate(data_name):
#     task_vecs[task_name] = task_vecs_one_hot[i]
# print('TASK VECS:', task_vecs)
    

'''
------------------------------------------------------------------------
学習
------------------------------------------------------------------------
'''
'''
出力結果
    max_acc：あるタスクにおける最高精度．{タスク名: 最高精度, ...}
    max_acc_epoch：あるタスクにおける最高精度を記録した時のepoch．{タスク名: 最高精度epoch, ...}
    max_avg_acc：複数タスク全体の最高精度．{タスク名: 最高精度, ...}
    max_avg_acc_epoch：複数タスク全体が最高精度を記録した時のepoch．int型

    avg_cost：あるepochにおける複数タスク全体の精度．{タスク名: 1x4リスト（[trainロス, train精度, valロス, val精度]）}
    cost：各batchでのコスト．1x2のリスト[ロス，精度]．
'''

max_acc = dict([(task_name, 0) for task_name in do_task_list])
max_acc_epoch = dict([(task_name, 0) for task_name in do_task_list])
max_avg_acc = dict([(task_name, 0) for task_name in do_task_list])
max_avg_acc_epoch = 0

# running
for index in range(load_epoch, total_epoch):
    avg_cost = dict([(task_name, [0,0,0,0]) for task_name in do_task_list])

    for task_name in do_task_list:
        
        cost = np.zeros(2, dtype=np.float32)
        train_dataset = iter(im_train_set[task_name])
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
                    task_vec = task_dict[task_name]['task_vec'].unsqueeze(0).repeat(batch,1).to(device)

                    train_pred1 = model(train_data, task_name, task_vec)
                    
                    train_loss1 = model.model_fit(train_pred1, train_label, num_output=task_dict[task_name]['num_class'], device=device)
                    train_loss = torch.mean(train_loss1)
                    train_loss.backward()
                    if args.random_lr:
                        optimizer.step(task_name=task_name)
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    # calculate training loss and accuracy
                    train_predict_label1 = train_pred1.data.max(1)[1]
                    train_acc1 = train_predict_label1.eq(train_label).sum().item() / train_data.shape[0]

                    cost[0] = torch.mean(train_loss1).item()
                    cost[1] = train_acc1
                    # avg_cost[index][task_name][0:2] += cost / train_batch
                    avg_cost[task_name][0:2] += cost / train_batch

            torch.save(model.state_dict(), 'weights/{}/{:0=5}.pth'.format(save_name, index+1))

                    
        if args.mode == 'train' or args.mode == 'val' or args.mode == 'test':
            # evaluating test data
            print('val')
            with torch.no_grad():
                model.eval()
                test_dataset = iter(im_test_set[task_name])
                test_batch = len(test_dataset)
                for i in range(test_batch):
                    test_data, test_label = test_dataset.next()
                    test_label = test_label.type(torch.LongTensor)
                    test_data, test_label = test_data.to(device), test_label.to(device)

                    batch = test_data.shape[0]
                    task_vec = task_dict[task_name]['task_vec'].unsqueeze(0).repeat(batch,1).to(device)

                    test_pred1 = model(test_data, task_name, task_vec, visualize=args.visualize)

                    test_loss1 = model.model_fit(test_pred1, test_label, num_output=task_dict[task_name]['num_class'], device=device)
                    test_loss = torch.mean(test_loss1)

                    # calculate testing loss and accuracy
                    test_predict_label1 = test_pred1.data.max(1)[1]
                    test_acc1 = test_predict_label1.eq(test_label).sum().item() / test_data.shape[0]

                    cost[0] = torch.mean(test_loss1).item()
                    cost[1] = test_acc1
                    # avg_cost[index][task_name][2:] += cost / test_batch
                    avg_cost[task_name][2:] += cost / test_batch
        
        if avg_cost[task_name][3] > max_acc[task_name]:
            max_acc[task_name] = avg_cost[task_name][3]
            max_acc_epoch[task_name] = index + 1
        print('EPOCH: {:04d} | DATASET: {:s} || TRAIN: {:.4f} {:.4f} || TEST: {:.4f} {:.4f} || MAX: {:.4f} ({:04d}ep)'
              .format(index+1, task_name,
                      avg_cost[task_name][0], avg_cost[task_name][1],
                      avg_cost[task_name][2], avg_cost[task_name][3],
                      max_acc[task_name], max_acc_epoch[task_name]))
    if sum(item[3] for item in avg_cost.values()) > sum(max_avg_acc.values()):
        for task_name in avg_cost.keys():
            max_avg_acc[task_name] = avg_cost[task_name][3]
            max_avg_acc_epoch = index + 1
    avg_acc_txt = '({:04d}ep)'.format(max_avg_acc_epoch)
    for task_name in do_task_list:
        avg_acc_txt += '{}: {:.4f}, '.format(task_name, max_avg_acc[task_name])
    print(avg_acc_txt)
    
    print('===================================================')

    if args.mode =='val' or args.mode == 'test': break
    
    # write tensorboardX
    train_loss = dict([(task_name, avg_cost[task_name][0]) for task_name in do_task_list])
    train_acc = dict([(task_name, avg_cost[task_name][1]) for task_name in do_task_list])
    val_loss = dict([(task_name, avg_cost[task_name][2]) for task_name in do_task_list])
    val_acc = dict([(task_name, avg_cost[task_name][3]) for task_name in do_task_list])
    writer.add_scalars('train_loss', train_loss, index+1)
    writer.add_scalars('train_acc', train_acc, index+1)
    writer.add_scalars('val_loss', val_loss, index+1)
    writer.add_scalars('val_acc', val_acc, index+1)
                
if args.mode=='train': writer.close()

