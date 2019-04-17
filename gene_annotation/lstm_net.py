import torch
from torch import nn
import torch
from torch import nn
import utils
import torch.nn.functional as F
import torch.autograd as autograd
import Config

class Lstm_Net(nn.Module):
    def __init__(self, sent_size):
        super(Lstm_Net, self).__init__()
        self.sent_size = sent_size
        self.embeds = nn.Embedding(len(utils.create_vocabulary(Config.window_size)), Config.embedding_size)
        self.embeds_size = Config.embedding_size * sent_size
        self.lstm = nn.LSTM(self.embeds_size, Config.hidden_layer_size, Config.num_layers)
        # self.fc1 = nn.Linear(self.embeds_size, Config.hidden_layer_size)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(Config.hidden_layer_size, Config.hidden_layer_size)
        self.relu = F.relu
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(Config.hidden_layer_size, 1)

    def forward(self, inputs):
        # import pdb; pdb.set_trace()
        embedding_weights = self.embeds(inputs).view((-1, self.embeds_size))
        embedding_weights.unsqueeze_(0)
        embedding_weights = embedding_weights.expand(1, -1, -1)
        lstm_output, (last_hidden_state, last_cell_state) = self.lstm(embedding_weights)
        layer1 = self.fc1(lstm_output[-1])
        layer1 = self.dropout(layer1)
        act1 = self.relu(layer1)
        layer2 = self.fc2(act1)
        layer2 = self.dropout(layer2)
        output = self.sigmoid(layer2)
        return output