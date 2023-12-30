import numpy as np
from torch.utils.data.dataset import Dataset as dataset

X_train = np.load("/content/gdrive/MyDrive/OOD_generalization/mate-mi-reg-model/DATA_2/X_TRAIN_DATA_CROSS_DOM_5_PERCENT.npy") # INPUT TRAIN DATA
print ("SHAPE TRAIN DATA:", X_train.shape)

Y_train = np.load("/content/gdrive/MyDrive/OOD_generalization/mate-mi-reg-model/DATA_2/Y_TRAIN_DATA_CROSS_DOM_5_PERCENT.npy") # TARGET INPUT TRAIN DATA
print ("Ytrain shape:", Y_train.shape)


class Dataset(dataset):
    def __init__(self, train=True, dom=0):
        super(Dataset, self).__init__()
        self.dom = dom

        if train:
            self.inputs = X_train  # inputs of all domains
            #self.cross_dom = X_train_cross_dom
            self.targets = Y_train
        else:
            pass

    def __getitem__(self, index):
        input = self.inputs[index]
        targets = self.targets[index]

        #return input, output, targets  ## input 3000 * 29 , output 1 * 3000 * 29
        return input, targets

    def __len__(self):
        return len(self.inputs)
    pass












