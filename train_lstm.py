import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle
import ipdb

from data import make_dataset, ReadData
from network import LSTMpred
from utils import to_variable, adjust_lr, save_checkpoint, AverageMeter

def main():
    # data 
    div1, div2 = 800, 900
    batch_size = 20
    num_workers = 0
    data_path = 'data/class1_data.pkl'
    # train
    num_epoch = 100
    lr = 1e-3
    lr_step = 50 # 
    momentum = 0.9
    weight_decay = 1e-3
    # model 
    hidden_dim = 32 # make it a number smaller than feature_dim
    model_check = 'model/checkpoint.pth.tar'
    model_best = 'model/bestmodel.pth.tar'
    # result
    print_freq = 20
    loss_best = 1e5


    # --------------------------------------------------------------------------
    # prepare dataset
    data, (len_seq, num_frame, num_joint, num_coor) = make_dataset(data_path)
    # data[:div] for trainning, data[800:900] for validation
    # and the rest for testing
    train_set = data[: div1]
    val_set = data[div1 : div2]
    # train_loader shuffle
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=1, 
                                shuffle=False, num_workers=num_workers)

    # --------------------------------------------------------------------------
    # model settings
    feature_dim = num_joint * num_coor # 16 * 3 = 48
    model = LSTMpred(feature_dim, hidden_dim)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum = momentum, 
                            weight_decay = weight_decay)

    # --------------------------------------------------------------------------
    # run
    
    for epoch in range(num_epoch):
        adjust_lr(lr, optimizer, epoch, lr_step)
        
        print('Epoch: {0}/{1} [training stage]'.format(epoch, num_epoch))
        train(train_loader, model, criterion, optimizer, print_freq)
        
        print('Epoch: {0}/{1} [validation stage]'.format(epoch, num_epoch))
        loss = val(val_loader, model, criterion, print_freq)
        
        is_best = loss < loss_best
        loss_best = min(loss_best, loss)
        save_checkpoint({
            'epoch': epoch,
            'arch': 'LSTMpred',
            'state_dict': model.state_dict(),
            'loss_best': loss_best,
            'optimizer' : optimizer.state_dict(),
        }, is_best, model_check, model_best)

    
# -----------------------------------------------------------------------------
# trainning

def train(train_loader, model, criterion, optimizer, print_freq):
    
    model.train()
    losses = AverageMeter()
    for step, (seq_in, seq_out) in enumerate(train_loader, 1):
        # ipdb.set_trace()
        seq_in = to_variable(seq_in)
        seq_out = to_variable(seq_out)
        
        # compute output
        model_out, r_out = model(seq_in)
        loss = criterion(model_out, seq_out)
        losses.update(loss.data[0], seq_in.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # show progress
        if step % print_freq == 0:
            print(r_out.data[0][0:9][:,0:10])
            print('Process: [{}/{}]\t Loss: {:.4f}'.format(
                step, len(train_loader), losses.avg))
  
    
# -----------------------------------------------------------------------------
# testing
def val(val_loader, model, criterion, print_freq):
    
    model.eval()
    losses = AverageMeter()
    for step, (seq_in, seq_out) in enumerate(val_loader, 1):
        
        seq_in = to_variable(seq_in)
        seq_out = to_variable(seq_out)
        
        # compute output
        model_out, r_out = model(seq_in)
        loss = criterion(model_out, seq_out)
        losses.update(loss.data[0], seq_in.size(0))
        
        # show progress
        if step % print_freq == 0:
            print('Process: [{}/{}]\t Loss: {:.4f}'.format(
                step, len(val_loader), losses.avg))
    return losses.avg  



if __name__ == '__main__':
	main()

