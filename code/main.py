import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader , Subset
from torchvision.transforms import v2
import torchvision.transforms.functional as F
from PIL import Image
import os
import random
import torch.optim as optim
from config import LEARNING_RATE , CLASS_WEIGHTS , DEVICE ,NUM_EPOCHS
from model import model
from utils import custom_weights
from data import TRAIN_DL , VALID_DL
import time

custom_weights()
LOSS_FN = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS).to(DEVICE))
OPTIMIZER = optim.Adam(model.parameters() , lr=LEARNING_RATE)


def train(model , n_epoch , train_dl , valid_dl , use_cuda = True):

    loss_hist_train = [0] * n_epoch
    loss_hist_valid = [0] * n_epoch
    acc_hist_valid = [0] * n_epoch
    acc_hist_train = [0] * n_epoch

    model.to(DEVICE)

    for epoch in range(n_epoch):
        start_time = time.time()

        model.train()

        for x , y in train_dl:

            x,y = x.to(DEVICE) , y.to(DEVICE).float()

            pred = model(x).squeeze(1)

            loss= LOSS_FN(pred , y)
            loss.backward()

            OPTIMIZER.step()
            OPTIMIZER.zero_grad()

            loss_hist_train[epoch] += loss.item() * y.size(0)
            is_correct = (torch.round(torch.sigmoid(pred)) == y).float()  # Apply sigmoid and round
            acc_hist_train[epoch] += is_correct.sum()

        loss_hist_train[epoch] /= len(train_dl.dataset)
        acc_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x, y in valid_dl:
                x, y = x.to(DEVICE), y.to(DEVICE).float()  # Convert target labels to float
                pred = model(x).squeeze(1)  # Remove the extra dimension from the output
                loss = LOSS_FN(pred, y)
                loss_hist_valid[epoch] += loss.item() * y.size(0)
                is_correct = (torch.round(torch.sigmoid(pred)) == y).float()  # Apply sigmoid and round
                acc_hist_valid[epoch] += is_correct.sum()
            loss_hist_valid[epoch] /= len(valid_dl.dataset)
            acc_hist_valid[epoch] /= len(valid_dl.dataset)


        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Training... Epoch {epoch+1}/{n_epoch} | Time: {epoch_time:.2f}s | "
              f"Train Accuracy: {acc_hist_train[epoch]:.4f} | Val Accuracy: {acc_hist_valid[epoch]:.4f}")

    return loss_hist_train, loss_hist_valid, acc_hist_train, acc_hist_valid

hist = train(model=model , n_epoch=NUM_EPOCHS , train_dl=TRAIN_DL , valid_dl= VALID_DL)

