#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:48:29 2017

@author: lei
"""

import numpy as np
import torch
import torch.utils.data as Data

traindata = np.load('/home/Lei/data/class200data.npy')
label = np.load('/home/Lei/data/class200label.npy')
traindata = traindata.astype('float32')
train_mean = np.mean(traindata)
traindata = traindata - train_mean
traindata = traindata.reshape(len(traindata), 1,64, 64)
#
print 'the size of class200:   ' + str(len(traindata))

train_x = torch.from_numpy(traindata)
label = torch.from_numpy(label)
data_torch = Data.TensorDataset(data_tensor= train_x, target_tensor= label)
trainloader = torch.utils.data.DataLoader(data_torch, batch_size = 16, shuffle = True)


testdata = np.load('/home/Lei/data/class200testdata.npy')
testlabel = np.load('/home/Lei/data/class200testlabel.npy')
testdata = testdata.astype('float32')
testdata = testdata - train_mean
testdata = testdata.reshape(len(testdata), 1,64, 64)

print 'the size of test:   ' + str(len(testdata))

test_x = torch.from_numpy(testdata)
testlabel = torch.from_numpy(testlabel)
testdata_torch = Data.TensorDataset(data_tensor= test_x, target_tensor= testlabel)
testloader = torch.utils.data.DataLoader(testdata_torch, batch_size = 100)


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,80,kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(80, 160, kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(160,320,kernel_size=3,padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
	self.conv4 = nn.Conv2d(320,640,kernel_size=3, padding=1)
        self.fc1   = nn.Linear( 640*4*4 , 1024)
        self.fc2   = nn.Linear(1024,200)
	self.drop  = nn.Dropout()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
       	x = self.pool3(F.relu(self.conv4(x)))
	x = x.view(-1, 640*4*4)
        x = F.relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
  #      x = F.softmax(x)
        return x

net = Net()
net.cuda()
net = nn.DataParallel(net)
#print(net)

from torch.nn.init import xavier_normal
from torch.nn.init import constant
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier_normal(m.weight.data)
        constant(m.bias.data,0)     
        
net.apply(weights_init)


import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[3,6,9,12 ], gamma=0.4)
lrer = StepLR(optimizer, step_size = 3, gamma=0.5)

print 'start training: '
for epoch in range(15): # loop over the dataset multiple times
    
    train_correct = 0
    train_acc=0.
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        scheduler.step()
       # get the inputs
        inputs, labels = data
        #inputs = inputs.view(-1,1,64,64)
        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        #inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        train_pred = torch.max(outputs,1)[1]
        
        train_correct = (train_pred == labels).sum()
        train_acc += train_correct.data[0]
        loss = criterion(outputs, labels)
        loss.backward()        
        optimizer.step()
       #	scheduler.step()
        # print statistics
        running_loss += loss.data[0]
        if i % 500 == 499: # print every 2000 mini-batches
            print('[%d, %5d] loss: %.5f' % (epoch+1, i+1, running_loss / 500))
            running_loss = 0.0
            
    print 'the trainset acc:  %.5f %%'  % (100 * float(train_acc) / len(data_torch))

    net.eval()        
    correct = 0
    total = 0
    for testdata in testloader:
        x, y = testdata
        x = x.cuda()
        y = y.cuda()
        o = net(Variable(x))
        _, predicted = torch.max(o.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum()

    print('the test acc: %.3f %%' % (100 * float(correct) / total))
print('Finished Training')














