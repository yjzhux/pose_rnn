import torch
from torch.autograd import Variable
import shutil

def to_variable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)

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

def adjust_lr(LR, optimizer, epoch, lr_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * (0.1 ** (epoch // lr_step))
    print('learning rate: ',lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename, save_best):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_best)  
        
# # remember best prec@1 and save checkpoint
# save_check = 'save/checkpoint.pth.tar'
# save_best = 'save/model_best.pth.tar'
# is_best = prec1 > best_prec1
# best_prec1 = max(prec1, best_prec1)
# print('best_prec1: ', best_prec1)
# save_checkpoint({
#     'epoch': epoch,
#     'arch': 'resnet101',
#     'state_dict': Model.state_dict(),
#     'best_prec1': best_prec1,
#     'optimizer' : optimizer.state_dict(),
# }, is_best, args.save_check)      