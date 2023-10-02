import torch
from utils import *
from torchvision import transforms
#from torch.utils.data.dataset import Dataset as dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset as dataset
from torch.utils.data import DataLoader

X_source = np.load("/content/gdrive/MyDrive/OOD_generalization/mate-mi-reg-model/DATA_2/X_TRAIN_IN.npy") # 3 * 1000 * 30 ## Ground truth source domain data
print ("SHAPE SOURCE TRAIN DATA:", X_source.shape)

X_train = np.load("/content/gdrive/MyDrive/OOD_generalization/mate-mi-reg-model/DATA_2/X_TRAIN_CROSS_DOM_20K.npy") # 3 * 1000 * 30 ## Train data #data_final_train ## data_fi
print ("SHAPE TRAIN DATA:", X_source.shape)


Y_train = np.load("/content/gdrive/MyDrive/OOD_generalization/mate-mi-reg-model/DATA_2/Y_TRAIN_CROSS_DOM_20K.npy") # 3000 * 1
print ("Ytrain shape:", Y_train.shape)


print ("X TRAIN SHAPE:", X_train.shape) # 20000 * 29 # stacked original data
print ("X SOURCE SHAPE:", len(X_source)) # 4 * 20000 * 29


class Dataset(dataset):
    def __init__(self, train=True, dom=0):
        super(Dataset, self).__init__()

        self.dom = dom

        if train:
            self.inputs = X_train  # inputs of all domains
            self.outs = X_source
            self.targets = Y_train
        else:
            pass


    def __getitem__(self, index):
        input = self.inputs[index]
        output = self.outs[index]
        targets = self.targets[index]

        return input, output, targets

    def __len__(self):
        return len(self.inputs)

    pass






















