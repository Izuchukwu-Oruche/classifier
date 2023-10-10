#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#                                                                             
# PROGRAMMER: Izuchukwu Oruche
# DATE CREATED: 10/09/2023                       

import argparse

def get_input_args():
    """
    PURPOSE:Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and define these command line arguments. If 
    the user fails to provide the optional arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Dataset Folder as --data_dir with default value 'flowers'
      2. Checkpoint save directory as --save_dir N.B: There is no default value
      3. CNN Model Architecture as --arch
      4. Model hyperparameters as --learning_rate, --hidden_units, --epochs
      5. GPU usage as --gpu
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = 'flowers/' , help = 'path to data directory')
    parser.add_argument('--save_dir', type = str, help = 'path to checkpoint directory', default='checkpoint_folder/')
    parser.add_argument('--arch', type = str, default = 'resnet', help = 'cnn model to use')
    parser.add_argument('--learning_rate', type= int, help= 'The learning rate of the algorithm', default = 0.003)
    parser.add_argument('--hidden_units', type = int, help = 'number of units in the hidden layer', default = 240)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default = 5)
    parser.add_argument('--gpu', type=str, default='cuda',help = 'choose whixh engine you want to train the network')
    parser.add_argument('--checkpoint_path', type=str, help='path to saved chekpoint', default= 'checkpoint_folder/model_checkpoint.pth')
    parser.add_argument('--path_to_image', type = str, help= 'Path to the image file')
    parser.add_argument('--topk', type = int, default=5, help= 'Number of top classes')
    parser.add_argument('--cat_name', type = str,default='cat_to_name.json', help= 'JSON file for the category names')


    return parser.parse_args()

