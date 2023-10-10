# This module contains functions and/or classes that will be used to process/analyse data

import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


def loading_data(data_dir):
    '''
    PURPOSE: From the data directory, creates datasets for training, and validation
    Performs transforms on the dataset so they can be operated on in PyTorch and creates Dataloaders
    where they can be loaded in batches to the model.
    Parameters:
    data_dir - A string object that represents the folder containing the entire dataset
    Returns:
    The training and validation data loaders as well as the class to index dictionary
    '''
    train_dir = data_dir + 'train/'
    valid_dir = data_dir + 'valid/'

    # transform definitions for the training, validation and testing datasets
    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.RandomRotation(30),
                                          transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    testing_validation_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), 
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # Using ImageFolder to load the datasets
    training_data = datasets.ImageFolder(train_dir, transform=training_transforms)
    
    validation_data = datasets.ImageFolder(valid_dir, transform=testing_validation_transforms)
    
    # Definition of the dataloaders
    trainloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)

    class_to_idx = training_data.class_to_idx
    

    return trainloader, validationloader, class_to_idx

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a NumPy array
    '''
    with Image.open(image_path) as im:
        shortest_side = 256
        width, height = im.size
        if width < height:
            new_width = shortest_side
            new_height = int(height * (shortest_side / width))
        else:
            new_width = int(width * (shortest_side / height))
            new_height = shortest_side
        im.thumbnail((new_width, new_height))
        width, height = im.size
        left = (width - 224) / 2
        top = (height - 224) / 2
        right = (width + 224) / 2
        bottom = (height + 224) / 2

        cropped_image = im.crop((left, top, right, bottom))
        print(type(cropped_image))
        normalised_image = np.array(cropped_image)/255

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized_image = (normalised_image - mean) / std
        transposed_image = normalized_image.transpose((2, 0, 1))
        return torch.tensor(transposed_image)

def predict(image_path, model, topk, engine):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    torch.cuda.empty_cache()
    image_tensor = process_image(image_path)
    image_tensor = image_tensor.float()
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        device = torch.device(engine)
        model.to(device)
        image_tensor = image_tensor.to(device)
        ps = torch.exp(model(image_tensor))
        ps = ps.cpu()
    top_ps, top_idx = ps.topk(topk)
    top_ps = top_ps.view(-1)
    top_ps = top_ps.numpy().tolist()
    top_idx = top_idx.view(-1)
    top_idx = top_idx.numpy().astype(int)
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_idx]
    
            
    return top_ps, top_classes


