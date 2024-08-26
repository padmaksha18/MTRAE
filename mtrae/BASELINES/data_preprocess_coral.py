import numpy as np
from torch.utils.data.dataset import Dataset as dataset

X_train = np.load("/content/gdrive/MyDrive/DATASETS/ARRYTHMIA/X_TRAIN_DATA_NORM_VEB_16K.npy", allow_pickle=True) # INPUT TRAIN DATA
print ("SHAPE TRAIN DATA:", X_train.shape)

#ADD CROSS DOMAIN DATA
X_train_cross_dom = np.load("/content/gdrive/MyDrive/DATASETS/ARRYTHMIA/X_TRAIN_CROSS_DOM__NORM_VEB_16K.npy") # CROSS DOM DATA INFIL, SOLARIS

Y_train = np.load("/content/gdrive/MyDrive/DATASETS/ARRYTHMIA/Y_TRAIN_DATA_NORM_VEB_16K.npy", allow_pickle=True) # TARGET INPUT #TRAIN DATA
print ("Ytrain shape:", Y_train.shape)


class Dataset(dataset):
    def __init__(self, train=True, dom=0):
        super(Dataset, self).__init__()
        self.dom = dom

        if train:
            self.inputs = X_train  # inputs of all domains
            self.cross_dom = X_train_cross_dom
            self.targets = Y_train
        else:
            pass

    def __getitem__(self, index):
        input = self.inputs[index]
        cross_dom = self.cross_dom[index]
        targets = self.targets[index]
        #print ("targets:", targets)

        #return input, output, targets  ## input 3000 * 29 , output 1 * 3000 * 29
        return input, cross_dom, targets

    def __len__(self):
        return len(self.inputs)
    pass
