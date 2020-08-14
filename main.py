import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================
# Modeling
class SequenceModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=256):
        super(SequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 4)

    def forward(self, x):
        zs, hidden = self.lstm(x)
        v = self.linear(zs)
        return v


# ======================================
# Prepare Data
data = pd.read_csv("./data.tsv", sep='\t', index_col=False)

window_size = 10
batch_size = 32
hidden_size = 256


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
n_epoch = 100

model = SequenceModel(input_size=4, hidden_size=hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

ema_loss = None
alpha = 0.1

for epoch_i in range(n_epoch):

    batch_list = make_batch(data, batch_size, window_size)
    for batch_i, batch in enumerate(batch_list):
        optimizer.zero_grad()

        batch = np.array(batch)
        batch_input = batch[:, :-1, :]
        batch_output = batch[:, 1:, :]

        batch_input = torch.tensor(batch_input, dtype=torch.float32)
        batch_output = torch.tensor(batch_output, dtype=torch.float32)

        v = model(batch_input)

        loss = loss_fn(v, batch_output)

        loss.backward()
        optimizer.step()

        if ema_loss is None:
            ema_loss = loss.item()
        ema_loss = loss.item() * alpha + (1.-alpha) * ema_loss

    print(f"{epoch_i}th epoch, loss: {loss.item()}")


# ======================================
# Inference

sample = [[5.6, 22.2,  1.0,  4.5],
          [5.0, 20.1,  1.0,  5.0],
          [4.7, 18.8,  1.0,  5.3],
          [3.2, 12.9,  1.0,  7.7],
          [4.2, 16.7,  1.0,  6.0],
          [7.2, 28.7,  1.0,  3.5],
          [6.0, 23.9,  1.0,  4.2],
          [3.7, 14.9,  1.0,  6.7],
          [2.3,  9.2,  1.0, 10.9],
          [6.4, 25.8,  1.0,  3.9]]

sample = np.array(sample)  # sequence_length x feature size
sample = torch.tensor(sample, dtype=torch.float32)  # sequence_length x feature size
sample = sample.unsqueeze(0)  # 1 x sequence_length x feature size

prediction = model(sample)
