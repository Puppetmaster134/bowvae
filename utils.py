
import numpy as np


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import torch
from torch.nn.functional import softmax




def compute_metrics(logits, labels, word_index):
    prob_dist = softmax(logits, dim=-1)
    results = torch.topk(prob_dist, 6, dim=-1)
    gen_bow = torch.zeros(labels.size())
    for index in range(results.indices.size(0)):
        gen_bow[index][results.indices[index]] = 1.0
    
    labels = labels.detach().cpu().numpy()
    preds = gen_bow.detach().cpu().numpy()
    correct_counts = []
    total_counts = []
    predictions = []
    actuals = []
    bleu_scores = []

    for idx, (label_bow, pred_bow) in enumerate(zip(labels,preds)):
        label_bow = np.flatnonzero(label_bow)
        pred_bow = np.flatnonzero(pred_bow)
        correct = np.intersect1d(label_bow, pred_bow)

        label_words = [word_index.idx2token(token_idx) for token_idx in label_bow.tolist()]
        pred_words = [word_index.idx2token(token_idx) for token_idx in pred_bow.tolist()]

        bleu_score = sentence_bleu([label_words], pred_words, weights=(1.0,))
        bleu_scores.append(bleu_score)

        #print(label_words)
        #print(pred_words)
        #print(bleu_score)
        #print()

        correct_counts.append(len(correct))
        total_counts.append(len(label_bow))
        predictions.append(pred_bow)
        actuals.append(label_bow)

    #print(np.mean(bleu_scores))

    return correct_counts, total_counts, actuals, predictions, np.mean(bleu_scores)