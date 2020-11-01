def adjust_learning_rate_and_learning_taks(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    if epoch >= args.step2: 
        lr = args.lr * 0.01
    elif epoch >= args.step1:
        lr = args.lr * 0.1
    else:
        lr = args.lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Return training classes
    return range(len(args.dataset))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count