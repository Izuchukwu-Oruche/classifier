import json
from get_input_arg import get_input_args
from model_functions import load_checkpoint
from utility_functions import predict
def main():
    in_arg = get_input_args() 
    # Import the category to flower name dictionary
    with open(in_arg.cat_name, 'r') as f:
        cat_to_name = json.load(f)

    

   

    model, optimizer, epochs = load_checkpoint(in_arg.checkpoint_path, in_arg.arch, in_arg.hidden_units, in_arg.learning_rate)

    top_ps, top_classes = predict(in_arg.path_to_image, model, in_arg.topk, in_arg.gpu)

    print('For the image file given, the top {} classes were {} with probabilities {}'.format(in_arg.topk, top_classes, top_ps))

    print('For the image file given, our trained model with {} percent confidence predicted the name of the flower was {}'.format(
        top_ps[0]*100, cat_to_name[top_classes[0]]))
    

if __name__ == "__main__":
    main()

