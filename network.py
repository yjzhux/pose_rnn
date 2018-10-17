import torch
import torch.nn as nn

class LSTMpred(nn.Module):

    def __init__(self, input_size, hidden_dim):
        super(LSTMpred,self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        # self.n_layer = n_layer
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=1, batch_first=True)
        # output_size = input_size
        self.hidden2out = nn.Linear(hidden_dim, input_size, bias=False)

    def forward(self,x):
        '''x shape (batch, time_step, input_size)
        r_out shape (batch, time_step, output_size)
        There are two hidden states in LSTM, namely h_n, h_c:
            h_n shape (n_layers, batch, hidden_size)   
            h_c shape (n_layers, batch, hidden_size)
        '''
        # None 表示 hidden state 会用全0的 state
        r_out, (h_n, h_c) = self.lstm(x, None) 
        model_out = self.hidden2out(r_out)
        # return outdat
        return model_out, r_out
