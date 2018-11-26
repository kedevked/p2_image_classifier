import argparse
import os

import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('dir', nargs='?', help='training data directory')
parser.add_argument('--save_dir', action='store',
                    default=os.getcwd(),
                    dest='directory',
                    help='directory to save checkpoints')

parser.add_argument('--arch', action='store',
                    default="vgg13",
                    dest='architecture',
                    help='model architecture')

parser.add_argument('--learning_rate', action='store',
                    default=0.01,
                    dest='lrate',
                    help='learning rate',
                    type=int
                   )

parser.add_argument('--hidden_units', action='store',
                    default=512,
                    dest='units',
                    help='number of hidden units',
                    type=int
                   )

parser.add_argument('--epochs', action='store',
                    default=20,
                    dest='epochs',
                    help='number of epochs',
                    type=int
                   )

parser.add_argument('--gpu', 
                    action='store_const', 
                    default=False, 
                    const=True,
                    help='use gpu')

arguments = parser.parse_args()

train_dir = arguments.dir

model = models[arguments.architecture](pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(1024, arguments.units)),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(arguments.units, 102))
                      ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier

device = 'gpu' if arguments.gpu and torch.cuda.is_available() else 'cpu'

def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0

data_transforms = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             ])

image_datasets = datasets.ImageFolder(arguments.dir, transform=data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets,batch_size=100, shuffle=True)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.lrate)    
do_deep_learning(model, dataloaders, 3, arguments.epochs, criterion, optimizer, device)

checkpoint = {'input_size': 1024,
              'output_size': 102,
              'state_dict': model.state_dict(),
              'class_to_idx': image_datasets.class_to_idx
             }

torch.save(checkpoint, arguments.directory)   