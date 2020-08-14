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
# Modeling
class SequenceModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_layers=2):
        super(SequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 4)

    def forward(self, x):
        zs, hidden = self.lstm(x)
        z = zs[:, -1]
        v = self.linear(zs)
        return v, z


# ======================================
# Prepare Data
data = pd.read_csv("./data.tsv", sep='\t', index_col=False)
data = data[:100]

window_size = 10
batch_size = 32
hidden_size = 128


def make_batch(data, batch_size, window_size):
    window_list = []
    for i in range(len(data) - window_size - 1):
        window = data[i: i + window_size]
        window_list.append(window)
    random.shuffle(window_list)

    n_batch = math.ceil(len(window_list) / batch_size)
    batch_list = []
    for i in range(n_batch):
        batch = window_list[i*batch_size: (i+1)*batch_size]
        batch_list.append(batch)
    batch_list = np.array(batch_list)

    return batch_list

data = data.to_numpy()


# ======================================
# Training
n_epoch = 10000

model = SequenceModel(input_size=4, hidden_size=hidden_size, num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

ema_loss = None
alpha = 0.1

for epoch_i in range(n_epoch):

    batch_list = make_batch(data, batch_size, window_size+1)
    for batch_i, batch in enumerate(batch_list):
        optimizer.zero_grad()

        batch = np.array(batch)
        batch_input = batch[:, :-1, :]
        batch_output = batch[:, 1:, :]

        batch_input = torch.tensor(batch_input, dtype=torch.float32)
        batch_output = torch.tensor(batch_output, dtype=torch.float32)

        v, _ = model(batch_input)

        loss = loss_fn(v, batch_output)

        loss.backward()
        optimizer.step()

        if ema_loss is None:
            ema_loss = loss.item()
        ema_loss = loss.item() * alpha + (1.-alpha) * ema_loss

    print(f"{epoch_i}th epoch, loss: {ema_loss}")


model.eval()

# ======================================
# Inference

# Pepare train data distribution
Z = []

batch_list = make_batch(data, batch_size, window_size)
for batch_i, batch in enumerate(batch_list):
    batch = np.array(batch)
    batch_input = batch

    batch_input = torch.tensor(batch_input, dtype=torch.float32)
    batch_output = torch.tensor(batch_output, dtype=torch.float32)

    _, z = model(batch_input)

    Z.extend(z.tolist())

Z = np.array(Z)


# Sample for quering
sample = [[5.1, 20.5,  1.0,  4.9],
          [4.1, 16.3,  1.0,  6.1],
          [9.1, 36.5,  1.0,  2.7],
          [2.3,  9.2,  1.0, 10.9],
          [1.6,  6.4,  1.0, 15.7],
          [6.6, 26.3,  1.0,  3.8],
          [8.0, 31.9,  1.0,  3.1],
          [7.8, 31.1,  1.0,  3.2],
          [7.0,  28.,  1.0,  3.6],
          [7.0,  28.,  1.0,  3.6]]

sample = np.array(sample)  # sequence_length x feature size
sample = torch.tensor(sample, dtype=torch.float32)  # sequence_length x feature size
sample = sample.unsqueeze(0)  # 1 x sequence_length x feature size

prediction, z_prime = model(sample)

z_prime = z_prime.detach().numpy()


dist_vector = pairwise_distances(z_prime, Z)



# ======================================
# Visualization

pca = PCA(n_components=2)
pca.fit(np.array(Z))

Z_2d = pca.transform(Z)

plt.scatter(Z_2d[:, 0], Z_2d[:, 1])
plt.show()




