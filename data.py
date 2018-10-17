import torch
from torch.utils.data import Dataset
import pickle

def make_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    (len_seq, num_frame, num_joint, num_coor) = data.shape # (931, 24, 16, 3)

    dat = list()
    for i in range(len_seq):
        # tranfer (24, 16, 3) --> (24, 48): [x0, y0, z0, x1, y1, z1, ...]
        seq = torch.from_numpy(data[i]).float().view(num_frame, num_coor * num_joint)
        indata = seq[0 : num_frame-1, :]
        outdata = seq[1 : num_frame, :]
        dat.append((indata, outdata))
    return dat, (len_seq, num_frame, num_joint, num_coor)


class ReadData(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(data)
        
    def __next__(self, idx):
        input, output = self.data[idx]
        return (input, output)

