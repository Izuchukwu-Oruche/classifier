# This module defines functions and/or classes pertaining to the model
import torch
from torch import nn
from torch import optim


from torchvision import models


def define_model(model_str, hidden_units):
    '''
    PURPOSE: Given a string to represent the model name, it defines that pretrained model that will be used for our problem
    Parameters:
    model_str - A string object that represents the model name
    Returns:
    The pretrained model
    '''
    # Building the pretrained network
    if model_str == 'vgg':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        fc = nn.Sequential(nn.Linear(model.fc.in_features, hidden_units),
                           nn.ReLU(), nn.Dropout(p=0.2),
                           nn.Linear(240, 102),
                           nn.LogSoftmax(dim=1))
        model.fc = fc

    if model_str == 'resnet':
        model = models.resnet34(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        fc = nn.Sequential(nn.Linear(model.fc.in_features, hidden_units),
                           nn.ReLU(), nn.Dropout(p=0.2),
                           nn.Linear(240, 102),
                           nn.LogSoftmax(dim=1))
        model.fc = fc 
    
    if model_str == 'densenet':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(nn.Linear(model.classifier.in_features, hidden_units),
                           nn.ReLU(), nn.Dropout(p=0.2),
                           nn.Linear(240, 102),
                           nn.LogSoftmax(dim=1))
        model.classifier = classifier
    




    # model = models.__dict__[model_str](weights = True)

    # # Freezing the cnn part of the network
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Definition of the classifier
    # fc = nn.Sequential(nn.Linear(model.fc.in_features, hidden_units),
    #                        nn.ReLU(), nn.Dropout(p=0.2),
    #                        nn.Linear(240, 102),
    #                        nn.LogSoftmax(dim=1))

    # model.fc = fc

    return model

def train_model(model,model_str, engine, learning_rate, epochs, trainloader, validationloader):
    '''
    PURPOSE: Given our pretrained model, this function trains the model for the given number of epochs and prints out the
    training loss, validation loss and validaton accuracy for each epoch
    Parameters:
    model - The pretrained model defined earlier
    engine -The engine used to train the model (CPU or GPU)
    learning_rate - An int object representing the learning rate of the algorithm
    epochs - An int object representing the number of epochs the algorithm will be trained for
    Returns:
    optimizer -to save as checkpoint
    '''
    torch.cuda.empty_cache()

    device = torch.device(engine)
    criterion = nn.NLLLoss()
    if model_str == 'vgg' or model_str == 'resnet':
        optimizer = optim.Adam(model.fc.parameters(), learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    
    model.to(device)
    
    cum_loss = 0


    for epoch in range(epochs):
        for images, labels in trainloader:

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()

        with torch.no_grad():
            model.eval()
            valid_loss = 0
            accuracy = 0

            for images, labels in validationloader:
                images, labels = images.to(device), labels.to(device)
                logps = model(images)
                loss = criterion(logps, labels)
                valid_loss += loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
            "Training Loss: {:.3f}.. ".format(cum_loss/len(trainloader)),
            "valid Loss: {:.3f}.. ".format(valid_loss/len(validationloader)),
            "valid Accuracy: {:.3f}".format(accuracy/len(validationloader))) 

        cum_loss = 0
        model.train() 

    return optimizer

def save_checkpoint(checkpoint_folder, model, class_to_idx, optimizer, epochs):
    '''
    PURPOSE: This function saves our trained model
    Parameters: 
    checkpoint_folder: Str object representing the folder where our checkpoint will be saved
    model: Our trained model
    class_to_idx: Dict object that maps the category classes to indices
    optimizer: The optimizer used to train our model
    epochs: Int object representing the number of epochs used to train our model
    Returns:
    None
    '''
    checkpoint_path = checkpoint_folder + 'model_checkpoint.pth'
    checkpoint ={'model_state_dict': model.state_dict(), 'class_to_idx':class_to_idx, 'optimizer_state_dict': optimizer.state_dict(), 'epochs': epochs}
    torch.save(checkpoint, checkpoint_path)

    return None

def load_checkpoint(checkpoint_path, model_str, hidden_units, learning_rate):
    '''
    PURPOSE: This function loads our model checkpoint
    Parameters:
    checkpoint_path: str object representing the path to the saved checkpoint
    model_str: str object representing the model
    hidden_units: int object representing the desired number of units in the hidden layer
    Returns:
    model - our model object
    optimizer - the optimizer
    epochs - Number of epochs used to train the model
    '''
    checkpoint = torch.load(checkpoint_path)
    model = define_model(model_str, hidden_units)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.fc.parameters(), learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epochs']
    return model, optimizer, epochs

    


