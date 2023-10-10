from utility_functions import loading_data
from get_input_arg import get_input_args
from model_functions import define_model, train_model, save_checkpoint





def main():

    # Get input arguments
    in_arg = get_input_args()    
      
    trainloader, validationloader, class_to_idx = loading_data(in_arg.data_dir)

    # Model definition including the classifier that will be trained for our problem
    model = define_model(in_arg.arch, in_arg.hidden_units)

    optimizer = train_model(model,in_arg.arch, in_arg.gpu, in_arg.learning_rate, in_arg.epochs, trainloader, validationloader)

    save_checkpoint(in_arg.save_dir, model, class_to_idx, optimizer, in_arg.epochs)

    

if __name__ == "__main__":
    main()








    


