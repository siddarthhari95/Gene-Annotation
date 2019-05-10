#Imports
import torch
from torch import nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import train
import Config
import utils
import test

def get_data(sample_file, sample_size):
    #return pd.read_csv(sample_file, header = None, nrows = sample_size)
    return pd.read_fwf(sample_file, sep = '\n', header = None, nrows = sample_size)

def data_shuffle(data, labels):
    return train_test_split(data, labels, test_size=Config.test_size, random_state=Config.seed)

if __name__ == '__main__':
    #Read from text files and create Train Data
    positive_train_data = get_data(Config.positive_train_sample_file, Config.positive_sample_size)
    negative_train_data = get_data(Config.negative_train_sample_file, Config.negative_sample_size)
    positive_train_data.columns = ["Gene"]
    negative_train_data.columns = ["Gene"]
    
    #Read from text files and create Test Data
    # positive_test_data = get_data(Config.positive_test_sample_file, Config.positive_test_sample_size)
    # negative_test_data = get_data(Config.negative_test_sample_file, Config.negative_test_sample_size)
    # positive_test_data.columns = ["Gene"]
    # negative_test_data.columns = ["Gene"]
    
    data_bef_shuf = positive_train_data.append(negative_train_data)
    labels_bef_shuf = utils.get_labels(Config.positive_sample_size, Config.negative_sample_size)

    # Shuffling the data
    train_data, test_data, train_labels, test_labels = data_shuffle(data_bef_shuf, labels_bef_shuf)

    #Typecasting labels to a torch tensor
    train_labels = torch.tensor(train_labels['label'].values, dtype=torch.float) #Cast the labels to type float tensor
    test_labels = torch.tensor(test_labels['label'].values, dtype=torch.float) #Cast the labels to type float tensor
    #labels_.type()

    # test_data = positive_test_data.append(negative_test_data)
    # test_labels = utils.get_labels(Config.positive_test_sample_size, Config.negative_test_sample_size)

    sent_size = len(positive_train_data.Gene[0]) - Config.window_size + 1

    train_accuracies, test_accuracies = train.train(train_data, train_labels, test_data, test_labels, sent_size)#Train the model

    test_accuracy = test.test(test_data, test_labels, sent_size, Config.model_name)
    print('Training accuracy for the trained model is: ', train_accuracies[len(train_accuracies)-1])
    print('Overall Test Accuracy is: ' , test_accuracy)
    if train_accuracies != None and test_accuracies != None:
        plt.xlabel('Epochs')
        plt.ylabel('Accuracies')
        plt.plot(train_accuracies)
        plt.plot(test_accuracies)
        plt.gca().legend(('Training', 'Test'))
        
        title = 'Test and Training Accuracies vs Epochs for: ' + (str)(Config.positive_sample_size + Config.negative_sample_size) + ' data points and ' + (str)(Config.num_epochs) + ' epochs'
        plt.title(title)
        plt.show()