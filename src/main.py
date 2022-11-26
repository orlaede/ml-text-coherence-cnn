
import argparse
import os as os

from model.model import Model
from model.model_helper import ModelHelper
from utils.dataset import prepare_dataset

#from google.colab import drive

# %matplotlib inline
#drive.mount('/content/gdrive', force_remount=True)

# dataset_path = './data'
# os.makedirs('./' + 'data', exist_ok=True) 

def main(dataset_path='./data', get_data='nogetdata', balanced='nobalanced'):
    if get_data == 'getdata':
        prepare_dataset()

    if balanced == 'balanced':
        make_balanced = True
    else:
        make_balanced = False

    model = Model()
    # make_balanced set balanced class adjust for dataset
    model.load_data_from_csv(data_path=dataset_path+'/dataset.csv', make_balanced=make_balanced)
    ModelHelper.train_model_kfolds(model.data, 
                                Model, 
                                ModelHelper.negative_log_likelihood, 
                                m.num_of_folds, 
                                plot_roc=True)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a text coherence CNN model')
    parser.add_argument('--dataset_path', metavar='path', required=True,
                        help='The path to dataset prepared (default: /data)')
    parser.add_argument('--get_data', required=True,
                        help='Inform to get data from web (getdata/nogetdata')
    parser.add_argument('--balanced', required=True,
                        help='Define to make dataset balanced (balanded/nobalanced')
    
    args = parser.parse_args()
    main(dataset_path=args.dataset_path, get_data=args.get_data, balanced=args.balanced)