import sys
sys.path.append(r'../')

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader


from textutils import tokenize

class BoWDataset(Dataset):
    def __init__(self, sentences, index, top_n = 6):
        self.data = sentences
        self.index = index
        self.top_n = top_n

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx].sentence

        try:
            tfidf_scores = self.index.tfidf(sentence, self.top_n)
        except Exception as e:
            raise Exception(f'[{idx}] - {e}')

        input_tensor = torch.zeros(len(self.index.n_tokens))

        for token in tfidf_scores.keys():
            input_tensor[self.index.token2idx(token)] = 1

        """
        if idx % 1000 == 0:
            print('TC ',tfidf_scores.keys())
        """
        return input_tensor

class BoWIndex():
    def __init__(self, df):
        self.data = df

        print(f'Indexing {len(self.data)} sentences.')

        self.UNK = '<UNK>'

        self.min_occurence = 0

        dict_count = { }

        for i, row in df.iterrows():
            tokens = list(set(tokenize(row.sentence)))

            for token in tokens:
                if token in dict_count.keys():
                    dict_count[token] = dict_count[token] + 1
                else:
                    dict_count[token] = 1
        

        n_singular_token = sum(map((1).__eq__, dict_count.values()))
        print('Singular Tokens: ',n_singular_token)

        dict_count = {token: n_occurence for token, n_occurence in dict_count.items() if n_occurence > self.min_occurence}
        dict_count[self.UNK] = n_singular_token
        self.n_tokens = dict_count
        self.tokenidx = { token : j for j, token in enumerate(dict_count.keys())}
        self.idxtoken = { j : token for token, j in self.tokenidx.items()}

        print(f'Unique tokens: {len(self.n_tokens)}')
    
    def __len__(self):
        return len(self.data)
    
    def token2idx(self, token):
        return self.tokenidx[token] if token in self.tokenidx.keys() else self.tokenidx[self.UNK]

    def idx2token(self, idx):
        return self.idxtoken[idx]
    
    def token_count(self, token):
        return self.n_tokens[token] if token in self.n_tokens.keys() else self.min_occurence
    
    def tfidf(self, sentence, top_n):
        tokenized_sentence = tokenize(sentence)
        if len(tokenized_sentence) == 0:
            raise Exception(f"Attempted to calculate tfidf for sentence of token length 0.\nSentence: {sentence}")
            
        
        tfidf_scores = {}
        distinct_tokens = list(set(tokenized_sentence))

        for token in distinct_tokens:
            tf = tokenized_sentence.count(token) / len(tokenized_sentence)
            idf = np.log(len(self.data) / (self.token_count(token) + 1))
            tfidf_scores[token] = tf * idf
        
        tfidf_scores = {k: v for k, v in sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]}
        return tfidf_scores
