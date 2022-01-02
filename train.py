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
    '''Defines the arguments available to call the script train.py'''
    parser = argparse.ArgumentParser('Arguments for the test.py script')
    #parser.add_argument('--gpu', type= bool, default=True, help='True for training with a GPU, False for training with just the CPU, defaults to True.')

    parser.add_argument('--gpu', type= str, default='cuda', help='True for training with a GPU, False for training with just the CPU, defaults to True.')
    parser.add_argument('--data_dir', type= str, default='flowers', required = True, help='Directory for data to be used to train our network, defaults to the flowers directory.')
    parser.add_argument('--epochs', type= int, default=3, help='Number of epochs to train our model, defaults to 3.')
    parser.add_argument('--learning_rate', type= float, default=0.001, help='Learning rate to train our model, defaults to 0.001.')
    parser.add_argument('--hidden_units', type= int, default=512, help='Number of hidden units for our model, defaults to 512')
    #make an argument for model architecture
    parser.add_argument('--arch', type= str, default='vgg16', help='Architecture for Classifier model, defaults to VGG16')
    args = parser.parse_args()
    return args

def load_data(data_dir): 
    '''Loads data for training, validation, and testing and applies the corresponding transforms for optimizing our model's performance'''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    training_transforms = transforms.Compose([transforms.RandomRotation(180),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],
                                                              [0.229,0.224,0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],
                                                              [0.229,0.224,0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])
    train_dataset = datasets.ImageFolder(train_dir, transform = training_transforms) 
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 64)
    
    return train_dataloader, test_dataloader, validation_dataloader, train_dataset
              
def construct_classifier(device, arch, hidden_units, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() and device =='gpu' else "cpu") 
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
    else:
        model = models.resnet18(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    #my new classifier 
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('relu', nn.ReLU()),
            ('Dropout1', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(4096, hidden_units)),
            ('Dropout2', nn.Dropout(p=0.2)), 
            ('fc3', nn.Linear(512,102)),
            ('Dropout3', nn.Dropout(p=0.2)), 
            ('output', nn.LogSoftmax(dim =1))
            ]))
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    return model, criterion, optimizer
    
def train_classifier(epochs,train_dataloader, device, model, criterion, optimizer, validation_dataloader):
    steps = 0 
    running_loss = 0
    print_every = 10
    model.to(device)
    for epoch in range(epochs):
        for images, labels in train_dataloader:
            steps+=1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() #Always zero out gradient
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validation_dataloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        loss = criterion(logps, labels)
                        test_loss += loss.item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                '''
                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Train Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(validation_dataloader)),
                      "Test accuracy: {:.3f}".format(accuracy/len(validation_dataloader)))
                      '''
                running_loss = 0
                model.train()
def test_classifier(model, device, criterion, optimizer, test_dataloader, train_dataset, epochs):
    model.to(device)
    loss = 0
    test_accuracy = 0
    steps = 0
    with torch.no_grad():
        for inputs_2, labels_2 in test_dataloader:
            steps += 1
            inputs_2, labels_2 = inputs_2.to(device), labels_2.to(device)
            logps = model.forward(inputs_2)
            batch_loss = criterion(logps, labels_2)
            loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels_2.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print("Test accuracy: {:.3f}".format(test_accuracy/len(test_dataloader)))
    # TODO: Save the checkpoint 
    model.class_to_idx = train_dataset.class_to_idx
    model_checkpoint = {'epochs':epochs,
                    'state_dict': model.state_dict(),
                    'class_to_idx':model.class_to_idx,
                    'optimizer':optimizer.state_dict(),
                    'classifier':model.classifier}
    torch.save(model_checkpoint, 'checkpoint.pth')
    
def main():
    args = get_input_args()
    data_dir = args.data_dir
    device = args.gpu
    epochs = args.epochs
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    arch = args.arch
    train_dataloader, validation_dataloader, test_dataloader, train_dataset = load_data(data_dir)
    model, criterion, optimizer = construct_classifier(device, arch, hidden_units, learning_rate)
    #DONT FORGET TO UNCOMMENT TRAINING PROGRES BEFORE FINAL SUBMISSION
    train_classifier(epochs, train_dataloader, device, model, criterion, optimizer, validation_dataloader) 
    test_classifier(model, device, criterion, optimizer, test_dataloader, train_dataset, epochs)
if __name__ == "__main__":
    main()
    
    

    
