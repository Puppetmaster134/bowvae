import sys
sys.path.append(r'../')

import pandas as pd
import numpy as np
import torch
import pickle

from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


from BoWDataset import BoWDataset, BoWIndex
from BoWVAE import BoWVAE

from utils import compute_metrics

use_cuda = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 32

output_dir = './saved'
data_dir = '../data/processed'
prompts_file = 'data_prompts_n10.csv'
sentences_file = 'data_sentences_n10.csv'

df_prompts = pd.read_csv(f'{data_dir}/{prompts_file}')
df_sentences = pd.read_csv(f'{data_dir}/{sentences_file}')

print(f'Number of Stories: {len(df_prompts)}')
print(f'Number of Sentences: {len(df_sentences)}')

index = BoWIndex(df_sentences)
pickle.dump(index, open(f"{output_dir}/word_index.p", "wb"))


df_train, df_val = train_test_split(df_sentences,test_size=0.05)
ds_train = BoWDataset(df_train,index)
dl_train = DataLoader(ds_train,batch_size=batch_size,shuffle=True, num_workers=0, pin_memory=True)

print(f'Validation Set Size: {len(df_val)}')

ds_val = BoWDataset(df_val,index)
dl_val = DataLoader(ds_val,batch_size=16, num_workers=0, pin_memory=True)

hidden_size = 256
latent_size = 128
n_vocab = len(ds_train.index.n_tokens)
temperature = 1.0

model = BoWVAE(hidden_size, latent_size, n_vocab, temperature, device=device).to(device)
adam = Adam(model.parameters(),lr=1e-3)

def loss_fn(logits, x):
    return -torch.sum(logits * x, dim=-1)

bce_logits = loss_fn

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - torch.exp(2 * logvar))

def kl_weight(step, n_batches, threshold_epoch):
    threshold = n_batches * threshold_epoch
    weight = np.round((np.tanh((step - threshold) / n_batches) + 1) / 2, decimals=6)

    return weight if weight < .03 else .03

def train(model, dataloader, optimizer, criterion, epoch):
    model.train()
    print_every = 500

    reconstruction_losses = []
    total_kl_loss = 0.0

    n_batches = len(dataloader)
    step = (epoch * n_batches)

    for i, batch in enumerate(dataloader):

        batch = batch.to(device)
        #Forward pass
        mean, logv, z, logits = model(batch)

        #Calculate loss
        reconstruction_loss = criterion(batch,logits)
        kld_loss = kl_divergence(mean,logv)
        weight = kl_weight(step, n_batches, 5)
        batch_loss = torch.mean(reconstruction_loss) + ((kld_loss * weight) / batch.shape[0])
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        reconstruction_losses.extend(reconstruction_loss.detach().cpu().tolist())
        total_kl_loss += kld_loss


        if step % print_every == 0:
            print('KL Weight: ', weight)
            #precision = np.sum(tokens_ret) / (len(tokens_ret) * config.sample_count)
            #recall = np.sum(tokens_ret) / np.sum(total_tokens)
            #logger.info(f'Epoch: {epoch} Step: {step+1:04} KL Weight: {KL_weight:0.4f} KL loss: {KL_loss.item() / batch_size:.4f}, Reconstruction loss: {rec_loss.item()/batch_size:.6f}, Total loss: {batch_loss.item():.6f}, Bleu Score: {np.mean(bleu_scores):.8f}')
            print(f'Epoch: {epoch} Step: {step} Reconstruction loss: {batch_loss.item():.4f}, KL (Batch Average): {kld_loss.item()/batch.shape[0]:.4f}')
            #print(torch.sum(batch,dim=1)[:5])
            #print(torch.sum(logits,dim=1)[:5])
        
        step = step + 1
    
    return np.mean(reconstruction_losses), total_kl_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion):
    model.eval()
    reconstruction_losses = []
    accuracies = []
    bleu_scores = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):         
            batch = batch.to(device)
            mean, logv, z, logits = model(batch)
            reconstruction_loss = criterion(batch,logits)
            reconstruction_losses.extend(reconstruction_loss.detach().cpu().tolist())
            num_correct, num_total, refs, preds, avg_bleu = compute_metrics(logits, batch, index)
            precision = np.sum(num_correct) / (len(num_correct) * 6)
            recall = np.sum(num_correct) / np.sum(num_total)
            num_correct = np.array(num_correct)
            num_total = np.array(num_total)
            max_correct = np.min(np.column_stack((num_correct, num_total)), axis=1)
            batch_accuracy = np.divide(max_correct,num_total).tolist()
            accuracies.extend(batch_accuracy)
            bleu_scores.append(avg_bleu)

    return np.mean(reconstruction_losses), np.mean(accuracies), np.mean(bleu_scores)



def train_epochs(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    step = 0

    val_accs = []
    val_bleu_scores = []
    highest_bleu = 0
    highest_acc = 0

    for epoch in range(num_epochs):
        print(f'Train Epoch {epoch + 1}')
        train_reconstruction_loss, train_kl_loss = train(model, train_loader, optimizer, criterion, epoch)

        print(f'Validation Epoch {epoch + 1}')
        val_reconstruction_loss, val_accuracy, val_bleu = validate(model, val_loader, criterion)

        print(f'Validation Accuracy: {val_accuracy}')
        print(f'Validation BLEU: {val_bleu}')

        if val_bleu > highest_bleu:
            highest_bleu = val_bleu
            print(f'New best Validation BLEU of {val_bleu}, saving model.')
            torch.save(model.state_dict(), f'{output_dir}/best_bowvae.pt')

        val_accs.append(val_accuracy)
        val_bleu_scores.append(val_bleu)
    
    plt.plot(range(1,len(val_accs) + 1),val_accs)
    plt.plot(range(1,len(val_accs) + 1),val_bleu_scores)
    plt.ylabel('acc/bleu')
    plt.xlabel('epoch')
    plt.savefig(f'{output_dir}/validation.png')




epochs = 20
train_epochs(model, dl_train, dl_val, adam, bce_logits, epochs)