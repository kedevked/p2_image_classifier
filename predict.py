import argparse
import os

import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from PIL import Image
import os
from torch.autograd import Variable

parser = argparse.ArgumentParser()

parser.add_argument('image', nargs='?', dest='image', help='image path')

parser.add_argument('dir', nargs='?', dest='checkpoint', help='checkpoint file')

parser.add_argument('--topk', action='store',
                    default=5,
                    dest='topk',
                    help='topk classes',
                    type=int
                   )

parser.add_argument('--category_names', action='store',
                    dest='categories',
                    help='json file with all categories',
                   )

parser.add_argument('--gpu', 
                    action='store_const', 
                    default=False, 
                    const=True,
                    help='use gpu')

arguments = parser.parse_args()

device = 'gpu' if arguments.gpu and torch.cuda.is_available() else 'cpu'

def load_checkpoint(model):
    model = models.densenet121(pretrained=True)
    
    classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(1024, 300)),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(300, 102))
                      ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

loaded_model = load_model(arguments.checkpoint)
with open(arguments.categories, 'r') as f:
    cat_to_name = json.load(f)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    
    width, height = image.size   # Get dimensions
    print('w', width, 'h', height)
    new_width, new_height = 224, 224
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    image = image.crop((left, top, right, bottom))
    
    # im.crop(8, 8, 248, 248)
    np_image = np.array(image)
    np_norm =  ( np_image - np.array([0.485, 0.456, 0.406]) ) / np.array([0.229, 0.224, 0.225])
    return np_norm.transpose((-1, 0, 1))    
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    img = process_image(image)
    
    # move model and input to gpu if available
    model.to(device)
    img.to(device)

    model.eval()

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(Variable(torch.FloatTensor([img])))

    ps, indices = torch.exp(output).topk(5)
    index_to_class = {v: k for k, v in model.class_to_idx.items()}
    return ps.numpy().flatten().tolist(), [index_to_class[index] for index in indices.numpy().flatten().tolist()]

predict(arguments.image, loaded_model, arguments.topk)
