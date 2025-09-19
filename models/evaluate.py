import numpy as np
from sklearn import linear_model as sk_lm
from sklearn import metrics as sk_mtr
from sklearn import model_selection as sk_ms
from sklearn import multiclass as sk_mc
from sklearn import preprocessing as sk_prep
import torch
from torch import nn
from torch_geometric.data import Data
from typing import Dict
from tqdm import tqdm

"""
Model Evaluation
"""
class LogisticRegression(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        weight_decay: float,
        is_multilabel: bool,
        device,
    ):
        super().__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        self._optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=0.01,
            weight_decay=weight_decay,
        )
        self._is_multilabel = is_multilabel
        self._loss_fn = (
            nn.BCEWithLogitsLoss()
            if self._is_multilabel
            else nn.CrossEntropyLoss()
        )
        self._num_epochs = 1000
        self._device = device
        for m in self.modules():
            self.weights_init(m)

        self.to(self._device)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.fc(x)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.train()

        X = torch.from_numpy(X).float().to(self._device)
        y = torch.from_numpy(y).to(self._device)

        for _ in tqdm(range(self._num_epochs), desc="Epochs", leave=False):
            self._optimizer.zero_grad()
            pred = self(X)
            loss = self._loss_fn(input=pred, target=y)
            loss.backward()
            self._optimizer.step()

    def predict(self, X: np.ndarray):
        self.eval()
        with torch.no_grad():
            pred = self(torch.from_numpy(X).float().to(self._device))

        if self._is_multilabel:
            return (pred > 0).float().cpu()
        else:
            return pred.argmax(dim=1).cpu()
        
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes) 

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
    
class LogReg_fit(nn.Module):
    def __init__(self, ft_in, nb_classes, weight_decay, device):
        super(LogReg_fit, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self._optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=0.001,
            weight_decay=weight_decay,
        )
        self._loss_fn = nn.CrossEntropyLoss()
        self._num_epochs = 10000
        self._device = device

        for m in self.modules():
            self.weights_init(m)

        self.to(self._device)
        
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
    
    def fit(self, train_embs, train_lbls, test_embs, test_lbls):
        self.train()
        epoch_flag = 0
        epoch_win = 0
        best_acc = torch.zeros(1).to(self._device)

        for epoch in range(self._num_epochs):
            self._optimizer.zero_grad()
            logits = self(train_embs)
            loss = self._loss_fn(input=logits, target=train_lbls)
            loss.backward()
            self._optimizer.step()

            if (epoch+1)%100 == 0:
                self.eval()
                logits = self(test_embs)
                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                if acc >= best_acc:
                    epoch_flag = epoch+1
                    best_acc = acc
                    epoch_win = 0
                else:
                    epoch_win += 1
                if epoch_win == 10:
                    break
                
        return epoch_flag, best_acc.item()