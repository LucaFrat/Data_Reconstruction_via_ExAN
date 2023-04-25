""" Networks """

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import Attack_fig6.constants as cons
from torchsummary import summary


class FCNet_512(nn.Module):
    """ d0-512-K Network """
    
    def __init__(self, in_size, hdim, out_size):
        super(FCNet_512, self).__init__()

        self.fc1 = nn.Linear(in_size, hdim) 
        self.fc2 = nn.Linear(hdim, out_size) 

    def forward(self, x):
        
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        out = F.softmax(self.fc2(h))
        
        return out


def initialize_network(test_loader):

    _, (example_data, _) = next(enumerate(test_loader))

    in_size = np.prod(np.array(example_data[0].shape))
    out_size = cons.OUT_SIZE
    hdim = cons.HDIM
    layer_dims = [in_size, hdim, out_size]

    net_512 = FCNet_512(*layer_dims)

    optimizer = optim.SGD(net_512.parameters(), 
                          lr=cons.LEARNING_RATE,
                          momentum=cons.MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    if cons.SHOW:
        summary(FCNet_512(*layer_dims).to(cons.DEVICE), input_size=(in_size,))

    return net_512, optimizer, criterion, layer_dims


def train(net, epoch, optimizer, criterion, train_loader, layer_dims):
    """ Training the NN """

    net.train()

    train_loss = 0
    train_acc = 0
    grads = {}
    params = {}

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.view(-1).to(torch.int64)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()


        # compute and store gradients
        for index, param in enumerate(net.parameters()):
            if param.grad is not None:
                grads_current = param.grad.data.view(-1).clone().cpu()
                if index%2 == 0:
                    suff = 'w'
                    # I want the gradients to be matrices
                    grads_current = grads_current.reshape(layer_dims[index//2], layer_dims[index//2+1])
                    grads[f'Layer_{index//2}_{suff}'] = grads_current
                else:
                    suff = 'b'
                    grads[f'Layer_{index//2}_{suff}'] = grads_current
                params[f'Layer_{index//2}_{suff}'] = param


        # ATTENTION, ILLEGAL !!!
        # we will delete this part once "algo_B1" starts working
        # compute 'small_g' gradients (after softmax)
        for index, sample_class in enumerate(target):
            output[index][sample_class] += -1
        small_grads = output

        # stop after the first batch
        break

        # useless for the goal of the paper 
        optimizer.step()
        train_loss = loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).sum()

    return grads, small_grads, params



def test(net, criterion, test_loader):
    """ Testing the NN """

    net.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            test_loss += criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).sum()

    return test_loss/len(test_loader), test_acc