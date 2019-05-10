import random

positive_sample_size = 1500000
negative_sample_size = 1500000
positive_train_sample_file = 'positive_sample.txt'
negative_train_sample_file = 'negative_sample.txt'
window_size = 3
embedding_size = 5
num_epochs = 100
batch_size = 100
hidden_layer_size = 99
hidden_layer_size_2 = 99
num_layers = 2
with_lstm = True
learning_rate = 0.07
model_name = 'models/fc_with_lstm.pt'
test_model_name = 'models/fc_with_lstm_per_epoch.pt'
test_size = 0.25
seed = random.randint(1, 101)
dropout_rate = 0.1
