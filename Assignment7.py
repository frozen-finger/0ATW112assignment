from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import torch as torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
scaler = StandardScaler()
print(data['data'].shape)
print(data['target'].shape)
X = data['data']
# X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(data['target'], dtype=torch.long)
train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.3)
print(len(train_idx))
train_x = X[train_idx]
train_y = Y[train_idx]
test_x = X[test_idx]
test_y = Y[test_idx]

class MLP(nn.Module):
    def __init__(self, idim, hdim, outputdim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(idim, hdim)
        self.linear2 = nn.Linear(hdim, outputdim)
        # self.dropout = nn.Dropout(0.5)
        # self.norm1 = nn.BatchNorm1d(hdim)

    def forward(self, X):
        a1 = self.linear1(X)
        # a1_b = self.norm1(a1)
        # a1 = self.dropout(a1)
        z = F.relu(a1)
        a2 = self.linear2(z)
        return a2

mlp = MLP(30, 15, 2)
optimizer = optim.Adam(mlp.parameters(), betas=(0.8, 0.95), lr=0.01)
for epoch in range(500):
    optimizer.zero_grad()
    output = mlp(train_x)
    loss = F.cross_entropy(output, train_y)
    loss.backward()
    optimizer.step()
    print(loss)

pred = mlp(test_x)
print(pred.shape)
pred = F.softmax(pred, dim=1)
pred = torch.argmax(pred, dim=1)
print(pred)
print(test_y.data)
print(classification_report(test_y.data, pred.data, target_names=data['target_names']))
