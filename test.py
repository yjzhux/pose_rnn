import os
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
from utils import to_variable, adjust_lr, AverageMeter

# data
div = 900
batch_size = 1
num_workers = 0
data_path = 'data/class1_data.pkl'
# model 
hidden_dim = 32 # make it a number smaller than feature_dim
model_check = 'model/checkpoint.pth.tar'
model_best = 'model/bestmodel.pth.tar'
# result
save_path = 'data/pred_data.pkl'


# --------------------------------------------------------------------------
# prepare dataset
data, (len_seq, num_frame, num_joint, num_coor) = make_dataset(data_path)
test_set = data[div :]
# test_set = data[0: 30]
# train_loader shuffle
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
                            
# --------------------------------------------------------------------------
# model settings
feature_dim = num_joint * num_coor # 16 * 3 = 48
model = LSTMpred(feature_dim, hidden_dim)
model.eval()
# loading trained model
if os.path.isfile(model_best):
    print(("=> loading checkpoint '{}'".format(model_best)))
    checkpoint = torch.load(model_best)
    start_epoch = checkpoint['epoch']
    loss_best = checkpoint['loss_best']
    model.load_state_dict(checkpoint['state_dict'])
    print(("=> loss_best: '{}' (epoch {})"
          .format(loss_best, checkpoint['epoch'])))
else:
    print("=> no checkpoint found.")
print(model)
criterion = nn.MSELoss()


# --------------------------------------------------------------------------
# testing
pred_data = []
losses = AverageMeter()
progress = tqdm(test_loader)
for step, (seq_in, seq_out) in enumerate(progress, 1):
    seq_in = to_variable(seq_in)
    seq_out = to_variable(seq_out)
    # compute output
    model_out, r_out = model(seq_in)
    loss = criterion(model_out, seq_out)
    losses.update(loss.data[0], seq_in.size(0))
    # ipdb.set_trace()
    # output matrix
    out = model_out.view(-1, 16, 3)
    pred_data.append(out.data.numpy())    

# save output
print('Average loss on testing set: {:.4f}'.format(losses.avg))
with open(save_path, 'wb') as f:
    pickle.dump(pred_data, f)



