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
        
        self.ch_lrs_dict = {}
        self.ch_lrs = []
        count = 0
            
        for task_idx,task_name in enumerate(do_task_list):
            for group in self.param_groups:
                ch_lrs = []
                for p in group['params']:
                    if group['lambda'] == 1.0:
                        ch_lr = [lr] * p.shape[0]
                    elif group['lambda'] == 0.0:
                        ch_lr = [0] * p.shape[0]
                        ch_indexes = torch.linspace(0, p.shape[0], len(do_task_list)+1, dtype=int)
                        ch_lr[ch_indexes[task_idx]:ch_indexes[task_idx+1]] = [lr] * (ch_indexes[task_idx+1] - ch_indexes[task_idx])
                    else:
                        ch_lr = [0] * p.shape[0]
                        share_ch = int(p.shape[0] * group['lambda'])
                        ch_lr[:share_ch] = [lr] * share_ch
                        ch_indexes = torch.linspace(share_ch, p.shape[0], len(do_task_list)+1, dtype=int)
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

    
    
    
    
# --------------------------------------------------------

class Adam_c(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, task_num=1):
            
        super().__init__(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False)
        
        self.ch_lrs_list = []
        self.ch_lrs = []
        count = 0
            
        for task_idx in range(task_num):
            for group in self.param_groups:
                ch_lrs = []
                for p in group['params']:
                    if len(group['lr'])==1:
                        ch_lr = [group['lr'][0]] * p.shape[0]
                    else:
                        if task_idx==0:
                            ch_lr1 = [1] * (p.shape[0]//2)
                            ch_lr2 = [0] * (p.shape[0] - p.shape[0]//2)
                        elif task_idx==1:
                            ch_lr1 = [0] * (p.shape[0]//2)
                            ch_lr2 = [1] * (p.shape[0] - p.shape[0]//2)
                        ch_lr = ch_lr1 + ch_lr2
                    ch_lrs += [ch_lr] # 各グループのレイヤ数*各レイヤのチャネル数
                self.ch_lrs += [ch_lrs] # グループ数*各グループのレイヤ数*各レイヤのチャネル数
            self.ch_lrs_list += [self.ch_lrs] # タスク数*グループ数*各グループのレイヤ数*各レイヤのチャネル数
        
        self.state = defaultdict(lambda :defaultdict(dict))
        
    
#     def zero_grad(self):
#             r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
#             for group in self.param_groups:
#                 for p in group['params']:
#                     for ch_idx, q in enumerate(p):
#                         if q.grad is not None:
#                             q.grad.detach_()
#                             q.grad.zero_()
                    
    
    @torch.no_grad()  
    def step(self, closure=None, task_idx=0):
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
            for state_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    
                for ch_idx, q in enumerate(p):
                    step_size = self.ch_lrs_list[task_idx][group_idx][state_idx][ch_idx] / bias_correction1
#                     print(q.shape, exp_avg.shape, denom.shape)
                    q.addcdiv_(exp_avg[ch_idx], denom[ch_idx], value=-step_size)

        return loss