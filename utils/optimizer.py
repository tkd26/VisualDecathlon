import torch
from torch.optim import Optimizer, Adam, SGD
import numpy as np
import math
import random
from collections import defaultdict
    
    
class SGD_c(SGD):

    def __init__(self, params, lr=1e-4, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=False, do_task_list=[]):

        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=nesterov)
        
        self.do_task_list = do_task_list
        self.ch_lrs_dict = {}
        self.ch_lrs = []
        count = 0
            
        for task_idx,task_name in enumerate(self.do_task_list):
            for group in self.param_groups:
                ch_lrs = []
                for p in group['params']:
                    if group['lambda'] == 1.0:
                        ch_lr = [lr] * p.shape[0]
                    elif group['lambda'] == 0.0:
                        ch_lr = [0] * p.shape[0]
                        ch_indexes = torch.linspace(0, p.shape[0], len(self.do_task_list)+1, dtype=int)
                        ch_lr[ch_indexes[task_idx]:ch_indexes[task_idx+1]] = [lr] * (ch_indexes[task_idx+1] - ch_indexes[task_idx])
                    else:
                        ch_lr = [0] * p.shape[0]
                        share_ch = int(p.shape[0] * group['lambda'])
                        ch_lr[:share_ch] = [lr] * share_ch
                        ch_indexes = torch.linspace(share_ch, p.shape[0], len(self.do_task_list)+1, dtype=int)
                        ch_lr[ch_indexes[task_idx]:ch_indexes[task_idx+1]] = [lr] * (ch_indexes[task_idx+1] - ch_indexes[task_idx])
                        
                    ch_lrs += [ch_lr] # 各グループのレイヤ数*各レイヤのチャネル数
                self.ch_lrs += [ch_lrs] # グループ数*各グループのレイヤ数*各レイヤのチャネル数
            self.ch_lrs_dict[task_name] = self.ch_lrs # タスク数*グループ数*各グループのレイヤ数*各レイヤのチャネル数


    # 学習率を変える
    def update(self, lr):
        self.ch_lrs_dict = {}
        self.ch_lrs = []
        count = 0
            
        for task_idx,task_name in enumerate(self.do_task_list):
            for group in self.param_groups:
                ch_lrs = []
                for p in group['params']:
                    if group['lambda'] == 1.0:
                        ch_lr = [lr] * p.shape[0]
                    elif group['lambda'] == 0.0:
                        ch_lr = [0] * p.shape[0]
                        ch_indexes = torch.linspace(0, p.shape[0], len(self.do_task_list)+1, dtype=int)
                        ch_lr[ch_indexes[task_idx]:ch_indexes[task_idx+1]] = [lr] * (ch_indexes[task_idx+1] - ch_indexes[task_idx])
                    else:
                        ch_lr = [0] * p.shape[0]
                        share_ch = int(p.shape[0] * group['lambda'])
                        ch_lr[:share_ch] = [lr] * share_ch
                        ch_indexes = torch.linspace(share_ch, p.shape[0], len(self.do_task_list)+1, dtype=int)
                        ch_lr[ch_indexes[task_idx]:ch_indexes[task_idx+1]] = [lr] * (ch_indexes[task_idx+1] - ch_indexes[task_idx])
                        
                    ch_lrs += [ch_lr] # 各グループのレイヤ数*各レイヤのチャネル数
                self.ch_lrs += [ch_lrs] # グループ数*各グループのレイヤ数*各レイヤのチャネル数
            self.ch_lrs_dict[task_name] = self.ch_lrs # タスク数*グループ数*各グループのレイヤ数*各レイヤのチャネル数


    @torch.no_grad()
    def step(self, closure=None, task_name=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for state_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                if len(p.shape)==1:
                    p.unsqueeze(1)

                for ch_idx, (q, d_q) in enumerate(zip(p, d_p)):
                    
                    q.unsqueeze(0)
                    q.add_(d_q, alpha=-self.ch_lrs_dict[task_name][group_idx][state_idx][ch_idx])

        return loss