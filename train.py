import argparse
import pandas as pd
import torch
from model import VAE
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable


def prepare_data(data):
    DATA = data.T
    DATA = (DATA - DATA.mean()) / DATA.std()
    data = torch.tensor(DATA.values, dtype=torch.float)
    return data

def train(args):
    data_path = args.data
    data = pd.read_csv(data_path, index_col=0) 
    data = prepare_data(data) 
    batch_size = args.batch_size
    lr = args.lr
    n_epochs = args.n_epochs
    verbose = args.verbose
    model = VAE(input_size=data.shape[0], batch_size=batch_size) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        loss_sum = 0
        for i in range(data.shape[1] // batch_size):
            batch = Variable(data[:, i*batch_size: (i+1)*batch_size], requires_grad=False)
            optimizer.zero_grad()
            recon_batch = model(batch)
            loss = criterion(recon_batch, batch)
            loss.backward()
            optimizer.step()
            loss_sum += loss.data.numpy()
        if verbose:
            print('Epoch: {} - loss: {:.2f}'.format(epoch, loss_sum))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', help='path to data file') 
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='size of batch')
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float, help='learning rate')   
    parser.add_argument('--n_epochs', dest='n_epochs', default=10, type=int, help='number of learning epochs')
    parser.add_argument('--verbose', dest='verbose', default=False, type=bool, help='verbose')
    args = parser.parse_args()
    print(args)
    train(args) 
