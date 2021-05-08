import pandas as pd
import numpy as np
import torch
import pickle

from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

import matplotlib.pyplot as plt


from BoWDataset import BoWDataset, BoWIndex
from BoWVAE import BoWVAE

from utils import compute_metrics

use_cuda = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

output_dir = './saved'

index = pickle.load(open(f"{output_dir}/word_index.p", "rb"))

hidden_size = 256
latent_size = 128
n_vocab = len(index.n_tokens)
temperature = 1.0

model = BoWVAE(hidden_size, latent_size, n_vocab, temperature,device)
model.load_state_dict(torch.load(f'{output_dir}/best_bowvae.pt'))

model = model.to(device)
model.eval()



def logits2words(logits):
    prob_dist = softmax(logits, dim=-1)
    _,word_indices = torch.topk(prob_dist, 6, dim=-1)
    word_indices = word_indices.flatten().cpu().numpy()

    #print(word_indices)
    
    pred_words = [index.idx2token(token_idx) for token_idx in word_indices]

    return pred_words

with torch.no_grad():

    for idx in range(50):
        z = torch.randn(1, latent_size).to(device)
        logits = model.decode(z)

        print(logits2words(logits))


    print()
    print()
    sentence = "I was getting really tired of my dog eating my homework"
    tfidf_scores = index.tfidf(sentence,6)

    input_tensor = torch.zeros(len(index.n_tokens))

    for token in tfidf_scores.keys():
        input_tensor[index.token2idx(token)] = 1
    
    input_tensor = input_tensor.view(1,-1).to(device)

    _,_,z,logits = model(input_tensor)

    print('Sample Sentence: ', sentence)
    print('Extracted Keywords: ', list(tfidf_scores.keys()))
    print('Reconstructed Keywords: ', logits2words(logits))

