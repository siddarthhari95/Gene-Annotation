import torch
import time
import utils
import Config
import test
import lstm_net as net
from torch import nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from progress.bar import Bar

def train(train_inputs, train_labels, test_data, test_labels, sent_size):
    fcNet = net.Lstm_Net(sent_size)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(fcNet.parameters(), lr=Config.learning_rate, momentum=0.0)
    losses, train_accuracies, test_accuracies = train_model(train_inputs, train_labels, test_data, test_labels, fcNet, optimizer, criterion, sent_size, Config.with_lstm)
    if losses != None:
        plt.plot(losses)
        title = 'Loss vs Epochs for: ' + (str)(Config.positive_sample_size + Config.negative_sample_size) + ' data points and ' + (str)(Config.num_epochs) + ' epochs'
        plt.title(title)
        plt.show()
    return train_accuracies, test_accuracies

def train_epoch(model, inputs, labels, optimizer, criterion):
    model.train()
    losses = []
    j = 0
    correct, wrong = 0,0
    inputs = utils.generateInputs(inputs)
    labels_hat = []
    for i in range(0, len(inputs), Config.batch_size):
        data_batch = inputs[i:i + Config.batch_size]
        labels_batch = labels[i:i + Config.batch_size]
        data_batch = autograd.Variable(data_batch)
        labels_batch = autograd.Variable(labels_batch)
        optimizer.zero_grad()
        labels_batch_hat = model(data_batch)
        loss = criterion(labels_batch_hat, labels_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        optimizer.step()        
        losses.append(loss.data.numpy())
        labels_hat.append(labels_batch_hat)
        correct, wrong = utils.get_train_accuracy(labels_batch_hat, labels_batch, j-1, len(labels_batch), correct, wrong)
        
        

    #print('labels_hat size>', len(labels_hat))
    #correct, wrong = 2,1
    loss = sum(losses)/len(losses)
    # print("Loss >> ", loss)
    # print("Labels hat list >> ", labels_hat)
    return loss, tuple((correct, wrong))
    
def train_model(train_inputs, train_labels, test_data, test_labels, model, optimizer, criterion, sent_size, with_lstm):
    losses = []
    print('Training the model:')
    start_time = time.time()
    train_accuracies, test_accuracies = [], []
    #labels_hat = []
    #test_labels = utils.get_labels(Config.positive_test_sample_size, Config.negative_test_sample_size)
    bar = Bar('Processing', max=Config.num_epochs)
    for epoch in range(Config.num_epochs):  # loop over the dataset multiple times
        loss, acc = train_epoch(model, train_inputs, train_labels, optimizer, criterion)
        losses.append(loss)
        train_accuracy = 100*(acc[0]/(acc[0]+acc[1]))
        train_accuracies.append(train_accuracy)
        torch.save(model.state_dict(), Config.test_model_name)
        test_accuracy = test.test(test_data, test_labels, sent_size, Config.test_model_name)
        test_accuracies.append(test_accuracy)
        bar.next()
    bar.finish()
    torch.save(model.state_dict(), Config.test_model_name)
    print('Finished. Training took %.3f' %((time.time() - start_time)/60), 'minutes.')
    return losses, train_accuracies, test_accuracies