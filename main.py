import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

# ======================================
# Prepare Data
def make_batch(data, batch_size, window_size, shuffle=True):
    window_list = []
    for i in range(len(data) - window_size - 1):
        window = data[i: i + window_size]
        window_list.append(window)

    if shuffle:
        random.shuffle(window_list)

    n_batch = math.ceil(len(window_list) / batch_size)
    batch_list = []
    for i in range(n_batch):
        batch = window_list[i*batch_size: (i+1)*batch_size]
        batch_list.append(batch)
    batch_list = np.array(batch_list)

    return batch_list

data = pd.read_csv("./data.tsv", sep='\t', index_col=False)
data = data.to_numpy()

# ======================================
# Modeling
class SequenceModel(nn.Module):
    def __init__(self, input_size=4, output_dim=4, hidden_size=256, num_layers=1):
        super(SequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.scaler_bias = nn.Parameter(torch.ones(input_size, requires_grad=True))
        self.scaler = nn.Parameter(torch.ones(input_size, requires_grad=True))
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = (x + self.scaler_bias) * self.scaler
        zs, hidden = self.lstm(x)
        z = zs[:, -1]
        v = self.linear(zs)
        return v, z


# ======================================
# Training
window_size = 10
batch_size = 64
hidden_size = 128
use_cuda = True

model = SequenceModel(input_size=4,
                      output_dim=4,
                      hidden_size=hidden_size,
                      num_layers=1)

if use_cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

n_epoch = 100000000
ema_loss = None
alpha = 0.1
verbose_interval = 50

for epoch_i in range(n_epoch):

    batch_list = make_batch(data, batch_size, window_size+1)
    for batch_i, batch in enumerate(batch_list):
        optimizer.zero_grad()

        batch = np.array(batch)
        batch_input = batch[:, :-1, :]
        batch_output = batch[:, 1:, :]

        batch_input = torch.tensor(batch_input, dtype=torch.float32)
        batch_output = torch.tensor(batch_output, dtype=torch.float32)

        if use_cuda:
            batch_input = batch_input.cuda()
            batch_output = batch_output.cuda()


        v, _ = model(batch_input)

        loss = loss_fn(v, batch_output)

        loss.backward()
        optimizer.step()

        if ema_loss is None:
            ema_loss = loss.item()
        ema_loss = loss.item() * alpha + (1.-alpha) * ema_loss

    if epoch_i % verbose_interval == 0:
        print(f"{epoch_i}th epoch, loss: {ema_loss}")



# ======================================
# Inference
model.eval()
model.cpu()


# Prepare train data distribution
Z = []
reconstruction_error = []

batch_list = make_batch(data, batch_size, window_size, False)
for batch_i, batch in enumerate(batch_list):
    batch = np.array(batch)
    batch_input = batch

    batch_input = torch.tensor(batch_input, dtype=torch.float32)
    batch_output = torch.tensor(batch_output, dtype=torch.float32)

    v, z = model(batch_input)

    Z.extend(z.tolist())
    reconstruction_error.extend(torch.sum(torch.abs(v-batch_input), dim=[1,2]).detach().tolist())

Z = np.array(Z)
reconstruction_error = np.array(reconstruction_error)


# Samples for quering
sample_pos = [[5.1, 20.5,  1.0,  4.9],
              [4.1, 16.3,  1.0,  6.1],
              [9.1, 36.5,  1.0,  2.7],
              [2.3,  9.2,  1.0, 10.9],
              [1.6,  6.4,  1.0, 15.7],
              [6.6, 26.3,  1.0,  3.8],
              [8.0, 31.9,  1.0,  3.1],
              [7.8, 31.1,  1.0,  3.2],
              [7.0,  28.,  1.0,  3.6],
              [7.0,  28.,  1.0,  3.6]]


sample_neg = [[ 66, 267,   5,   0],
       	      [ 74, 298,   5,   0],
       	      [ 88, 354,   5,   0],
              [ 83, 335,   5,   0],
              [ 78, 315,   5,   0],
              [ 96, 385,   5,   0],
              [ 15,  59,   5,   1],
              [ 67, 267,   5,   0],
              [ 75, 303,   5,   0],
              [ 60, 242,   5,   0]]

# pos
sample_pos = np.array(sample_pos)  # sequence_length x feature size
sample_pos = torch.tensor(sample_pos, dtype=torch.float32)  # sequence_length x feature size
sample_pos = sample_pos.unsqueeze(0)  # 1 x sequence_length x feature size
prediction_pos, z_prime_pos = model(sample_pos)

# neg
sample_neg = np.array(sample_neg)  # sequence_length x feature size
sample_neg = torch.tensor(sample_neg, dtype=torch.float32)  # sequence_length x feature size
sample_neg = sample_neg.unsqueeze(0)  # 1 x sequence_length x feature size
prediction_neg, z_prime_neg = model(sample_neg)

z_prime_pos = z_prime_pos.detach().numpy()
z_prime_neg = z_prime_neg.detach().numpy()

reconstruction_error_pos = torch.sum(torch.abs(prediction_pos - sample_pos), dim=[1,2]).detach().tolist()
reconstruction_error_neg = torch.sum(torch.abs(prediction_neg - sample_neg), dim=[1,2]).detach().tolist()


# ======================================
# Visualize latent space
pca = PCA(n_components=2)
pca.fit(Z)

Z_2d = pca.transform(Z)

z_prime_pos_2d = pca.transform(z_prime_pos)
z_prime_neg_2d = pca.transform(z_prime_neg)

plt.scatter(Z_2d[:, 0], Z_2d[:, 1], color='k')
plt.scatter(z_prime_pos_2d[:, 0],z_prime_pos_2d[:, 1] , color='g', label='normal')
plt.scatter(z_prime_neg_2d[:, 0],z_prime_neg_2d[:, 1] , color='r', label='abnormal')
plt.legend()
plt.show()


# ======================================
# Plot Reconstruction Error 
neg_height = 50
min_val = min(min(reconstruction_error), min(reconstruction_error_neg))
max_val = max(max(reconstruction_error), max(reconstruction_error_neg))
bins = np.linspace(min_val, 
                   max_val,
                   100)

plt.hist(reconstruction_error_neg * neg_height, bins=bins, alpha=0.5, color='red', label='abnormal')
plt.hist(reconstruction_error, bins=bins, alpha=0.5,color='k', label='normal')
plt.legend()
plt.show()
