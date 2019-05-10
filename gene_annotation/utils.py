import torch
from torch import nn
import Config
import pandas as pd
import numpy as np

def get_labels(positive_sample_size, negative_sample_size):
    #labels = torch.cat((torch.ones([Config.positive_sample_size, 1,1], dtype=torch.float), torch.zeros([Config.negative_sample_size, 1,1], dtype=torch.float)))
    
    zeros = pd.DataFrame(np.zeros((negative_sample_size, 1), dtype=int))
    ones = pd.DataFrame(np.ones((positive_sample_size, 1), dtype=int))
    labels = pd.concat([ones, zeros])
    labels.columns = ['label']
    return labels

def embeddings_helper(window_size):
    vocab_set = set()

    def generate_vocab_helper(set, k): 
        n = len(set)  
        generate_vocab(set, "", n, k) 

    def generate_vocab(set, prefix, n, k): 
        if (k == 0) : 
            vocab_set.add(prefix)
            return
        for i in range(n): 
            newPrefix = prefix + set[i] 
            generate_vocab(set, newPrefix, n, k - 1) 

    def generate_embed_map(n):
        alphabet = ['0','1','2','3','4']
        generate_vocab_helper(alphabet, n)

        vocab_set_1 = sorted(vocab_set)
        vocab_map = {}

        for i in range(len(vocab_set_1)):
            vocab_map[vocab_set_1[i]] = i
        return vocab_map
    return generate_embed_map(window_size)

def return_embeddings(vocabulary):
    embeds = nn.Embedding(len(vocabulary), Config.embedding_size)
    embeddings = {}
    for word in vocabulary:
        embeddings[word] = embeds(torch.tensor(vocabulary[word], dtype=torch.long))
    return embeddings

#ATG, GTG, TTG
def is_start_codon(codon):
    start_codons = ['143', '343', '443']#['ATG', 'GTG', 'TTG']
    if codon in start_codons:
        return True
    return False

def create_vocabulary(window_size):
    return embeddings_helper(window_size)

def get_train_accuracy(labels_hat, labels, index, data_size, correct, wrong):    
    for i in range(0, len(labels_hat)):
        o = 0
        y_hat = labels_hat[i]
        o = 0 if ((float)(y_hat) <= 0.5) else 1
        if o == (int)(labels[i]):
            correct += 1
        else:
            wrong += 1
            
    # if index % 2 == 1:
    #     if label > 0.5:
    #         correct += 1
    #     else:
    #         wrong += 1
    # else:
    #     if label > 0.5:
    #         wrong += 1
    #     else:
    #         correct += 1
    return correct, wrong

def get_test_accuracy(labels_hat, labels):
    correct, wrong = 0, 0
    for i in range(0, len(labels_hat)):
        o = 0
        y_hat = labels_hat[i]
        o = 0 if ((float)(y_hat) <= 0.5) else 1
        if o == (int)(labels[i]):
            correct += 1
        else:
            wrong += 1
    accuracy = 100 * (correct/(correct+wrong))
    return accuracy

def generateInputs(inputs):
    #data = torch.tensor([], dtype=torch.float)
    data = []
    #print("inputs: ", len(inputs))
    vocabulary = create_vocabulary(Config.window_size)
    for input in inputs.itertuples():
        #print('>>', input)
        gene = input.Gene
        input_ = torch.tensor([vocabulary[gene[i:i+Config.window_size]] for i in range(0, len(gene) - Config.window_size + 1)], dtype=torch.long)
        # print("input>> ",input_)
        data.append(input_)
    # print(len(data))
    data = torch.stack(data)
    # for i in range(0, len(data), 10):
    #     print(data[i:i+10])
    #     print("---------------------------")
    return data