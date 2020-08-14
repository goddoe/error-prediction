import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


# ======================================
# Modeling
class SequenceModel(nn.Module):
    def __init__(self):
        super(SequenceModel, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=128, batch_first=True)
        self.linear = nn.Linear(128, 4)

    def forward(self, x):
        zs, hidden = self.lstm(x)
        v = self.linear(zs)
        return v


# ======================================
# Prepare Data
data = pd.read_csv("./data.tsv", sep='\t', index_col=False)

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
batch_list = make_batch(data, batch_size, window_size)


# ======================================
# Training
n_epoch = 100

model = SequenceModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


for epoch_i in range(n_epoch):
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

    print(f"{epoch_i}th epoch, loss: {loss.item()}")
