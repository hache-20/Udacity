# TODO: Write a function that loads a checkpoint and rebuilds the model
from PIL import Image
import json
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np

def get_input_args():
    '''Defines the arguments available to call the script predict.py'''
    parser = argparse.ArgumentParser('Arguments for the predict.py script')
    parser.add_argument('--GPU', type= bool, default=True, help='True for predicting with a GPU, False for training with just the CPU, defaults to True.')
    parser.add_argument('--img_path', type= str, default='"flowers/test/20/image_04912.jpg"', help='Directory for data to be used to test our network, defaults to the flowers test directory.')
    parser.add_argument('--top_k', type= int, default=5, help='Used for calculating the top k probabilities of a given image, defaults to the top 5')
    args = parser.parse_args()
    return args

def load_checkpoint(filepath):
    checkpoint =  torch.load(filepath)
    model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.required_grad = False
    model.classifier = model_checkpoint['classifier']    
    model.class_to_idx = model_checkpoint['class_to_idx']
    #model.state_dict = model_checkpoint['state_dict']
    model.load_state_dict(model_checkpoint['state_dict'])
    optimizer.load_state_dict(model_checkpoint['optimizer'])
    
    return optimizer, model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    image = image.resize((256,256))
    image = image.crop((0,0,224,224))
    np_image = np.array(image)
    np_image = np_image/255
    mean = np.array([0.485,0.456,0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std_dev
    np_image = np_image.transpose((2,1,0))
    
    return torch.from_numpy(np_image)

def predict(device, img_path, model, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.float()
    image = image.to(device)
    
    with torch.no_grad():
        output = model.forward(image)
        probabilities = torch.exp(output)
        top_p, top_classes = probabilities.topk(topk, dim=1)
        model.idx_to_class = dict(map(reversed,model.class_to_idx.items()))
        
    return top_p, top_classes

def main(): #populate
    args = get_input_args()
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    model = load_checkpoint(checkpoint)
    probs, classes = predict(img_path,model)
    print(probs, classes)
