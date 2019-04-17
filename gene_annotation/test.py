import torch
import lstm_net as net
import Config
import utils
from collections import OrderedDict

def test(inputs, labels, sent_size, model_name):
    model = net.Lstm_Net(sent_size)
    model.load_state_dict(torch.load(Config.test_model_name))
    
    model.eval()
    labels_hat = []
    inputs = utils.generateInputs(inputs)
    labels_hat = model(inputs)
    accuracy = utils.get_test_accuracy(labels_hat, labels)
    return accuracy