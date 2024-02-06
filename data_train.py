import torch, torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import pandas as pd
import shap

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

df = pd.read_csv('data_ml.csv', index_col=['datetime'], parse_dates=['datetime'])


n = len(df)

df_train, df_valid, df_test = df.iloc[:15000], df.iloc[15000:30000], df.iloc[30000:]

df_train = df_train.dropna(how='any')
df_valid = df_valid.dropna(how='any')
df_test = df_test.dropna(how='any')

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming the last column is the target
        X = self.data.iloc[idx, :-1].values
        y = self.data.iloc[idx, -1]

        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y

train_ds = CustomDataset(df_train)
valid_ds = CustomDataset(df_valid)
test_ds  = CustomDataset(df_test)


train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_ds, batch_size=32, shuffle=False)
test_dataloader  = DataLoader(test_ds, batch_size=32, shuffle=False)


class TinyModel(nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = nn.Linear(22, 100)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(100, 22)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(22, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x

# loss_fn = nn.MSELoss()
loss_fn = nn.L1Loss()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.inference_mode():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            test_loss += loss_fn(output, target).item() # sum up batch loss
    #        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    #        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, 0, len(test_loader.dataset),
    100. * 0 / len(test_loader.dataset)))



model = TinyModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

for epoch in range(1, 5):
    train(model, device, train_dataloader, optimizer, epoch)
    test(model, device, valid_dataloader)
