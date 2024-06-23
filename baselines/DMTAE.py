import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocess import Dataset as Dataset
# from autoencoder import Autoencoder
import numpy as np
import matplotlib.pyplot as plt
from utils import classify_with_knn
import pandas as pd

##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 300
batch_size = 200
feats = 20
domains = 10
classes = 2
latent_dim = 15
learning_rate = 0.005

# model = Autoencoder().to(device)
criterion = nn.MSELoss()


class MultitaskAutoencoder(nn.Module):
    def __init__(self, D_in, H=25, H2=20, latent_dim=15):
        # Encoder
        super(MultitaskAutoencoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H)  # 29 * 20
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2 = nn.Linear(H, H2)  # 20 * 10
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, H2)  # 10 * 10
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)
        self.num_class = classes

        self.fc1 = nn.Linear(H2, latent_dim)  # 10 * 7

        self.classifier = nn.Linear(latent_dim, self.num_class)  # 7 * 3

        # classifier
        # self.classifier = nn.Linear(latent_dim, self.num_class) # 7 * 3

        #         # Decoder
        self.fc3 = nn.Linear(latent_dim, latent_dim)  # 7 * 7
        #         self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)  # 7 * 10
        #         self.fc_bn4 = nn.BatchNorm1d(H2)

        self.linear4 = nn.Linear(H2, H2)  # 10 * 10
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H)  # 10 * 20
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6 = nn.Linear(H, D_in)  # 20 * 29
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)
        self.relu = nn.ReLU()

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))  # 29 * 20
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))  # 20 * 10
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))  # 10 * 10

        fc1 = self.relu(self.fc1(lin3))  # 10 * 7
        # fc2 = F.relu(self.classifier(fc1)) # 7 * 3
        return fc1

    def decode(self, z):
        fc3 = self.relu(self.fc3(z))  # 7 * 7
        fc4 = self.relu(self.fc4(fc3))  # .view(128, -1) # 7 * 10

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))  # 10 * 10
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))  # 10 * 20

        return self.lin_bn6(self.linear6(lin5))  # 20 * 29

    def forward(self, inputs):  # batch * feats

        z = self.encode(inputs)  # 29 * 3
        logits = self.classifier(z)
        reconstruction = self.decode(z)

        return logits, reconstruction, z


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # (reduction="sum")
        self.classification_criterion = nn.CrossEntropyLoss()

    def calculate_gram_mat(self, X, sigma):  # required only for codes
        """calculate gram matrix for variables x
            Args:
            x: random variable with two dimensional (N,d).
            sigma: kernel size of x (Gaussain kernel)
        Returns:
            Gram matrix (N,N)
        """
        x = X.view(X.shape[0], -1)
        instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
        dist = -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()

        return torch.exp(-dist / sigma)

    def renyi_entropy(self, code, sigma):  # code is batch * latent dim
        # calculate entropy for single variables x (Eq.(9) in paper)
        #         Args:
        #         x: random variable with two dimensional (N,d).
        #         sigma: kernel size of x (Gaussain kernel)
        #         alpha:  alpha value of renyi entropy
        #     Returns:
        #         renyi alpha entropy of x.

        alpha = 2  ## Renyi's 2nd order entropya

        # calculate kernel with new updated sigma
        code_k = self.calculate_gram_mat(code, sigma)
        code_k = code_k / torch.trace(code_k)
        # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
        eigv, eigvec = torch.linalg.eigh(code_k)
        eig_pow = eigv ** alpha
        entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
        # entropy = -torch.sum(eig_pow)

        return entropy

    def joint_entropy(self, code, prior, s_x, s_y):  # x = code (batch * feats), y = prior kernel (bacth * batch)

        """calculate joint entropy for random variable x and y (Eq.(10) in paper)
            Args:
            x: random variable with two dimensional (N,d).
            y: random variable with two dimensional (N,d).
            s_x: kernel size of x
            s_y: kernel size of y
            alpha:  alpha value of renyi entropy
        Returns:
            joint entropy of x and y.
        """

        alpha = 2

        code_k = self.calculate_gram_mat(code, s_x)
        prior_k = self.calculate_gram_mat(prior, s_y)

        # code_k = self.calculate_corr(code)
        # prior_k = self.calculate_corr(prior)

        k = torch.matmul(code_k, prior_k)
        k = k / torch.trace(k)
        # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
        eigv, eigvec = torch.linalg.eigh(k)
        eig_pow = eigv ** alpha
        entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
        # entropy = torch.sum(eig_pow)

        return entropy

    def entropy_loss(self, latent_code, prior_kernel, normalize):  ## calculate MI # x = code , y = prior

        """calculate Mutual information between random variables x and y
        Args:
            x: random variable with two dimensional (N,d).
            y: random variable with two dimensional (N,d).
            s_x: kernel size of x
            s_y: kernel size of y
            normalize: bool True or False, noramlize value between (0,1)
        Returns:
            Mutual information between x and y (scale)

        """
        # global s_x
        s_x = 0.2  # code
        s_y = 0.5  # prior

        # entropy of code. code is batch * latent dimension
        Hx = self.renyi_entropy(latent_code, sigma=s_x)

        # entropy of prior ##For prior, RBF kernel is pre-computed. sigma is not considered
        Hy = self.renyi_entropy(prior_kernel, sigma=s_y)

        # joint entropy
        # Hxy = joint_entropy(x, y, s_x, s_y)
        Hxy = self.joint_entropy(latent_code, prior_kernel, s_x, s_y)

        if normalize:
            Ixy = Hx + Hy - Hxy
            # Ixy = ((Hx * Hy) / (Hxy * Hxy))
            # Ixy = -Hx - Hy + (2 * Hxy)
            Ixy = Ixy / (torch.max(Hx, Hy))
        else:
            Ixy = Hx + Hy - Hxy
            # Ixy = ((Hx * Hy) / (Hxy * Hxy))
            # Ixy = -Hx - Hy + (2 * Hxy)
            Ixy = Ixy / (torch.max(Hx, Hy))

        return Ixy

    def forward(self, input, dec_out, Z, dom_out, logits, targets):
        loss_MSE = self.mse_loss(dec_out, dom_out)
        # print ("LOSS MSE:", loss_MSE)

        targets = torch.flatten(targets)
        # print ("TARGETS:",targets)

        #classification_loss = self.classification_criterion(logits, targets)

        #entropy_loss = self.entropy_loss(input, Z, True)

        total_loss = loss_MSE

        #total_loss = classification_loss + (0.9 * loss_MSE) #+ (2 * entropy_loss)

        # return classification_loss + (0.5 * loss_MSE) + (1 * entropy_loss)
        return total_loss


# D_in = data_set.x.shape[1]
D_in = 29
print("D_in shape:", D_in)
H = 25
H2 = 20

log_interval = 150
val_losses = []
train_losses = []

multitaskAE = MultitaskAutoencoder(D_in, H, H2).to(device)
optimizer = torch.optim.Adam(multitaskAE.parameters(), lr=learning_rate, weight_decay=1e-5)


def train_AE():
    for epoch in range(epochs):
        train_loss = 0
        losses = []

        for domain in range(domains):
            dataset = Dataset(dom=domain)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            for input, out, targets in dataloader:
                # data = data.to(device)

                input = input.float().to(device)
                # print ("INPUTS SHAPE:", input.shape)

                dom_out = out.float().to(device)  # ground truth data
                # print("OUTS SHAPE:", dom_out.shape)

                targets = targets.long().to(device)
                # print ("TARGETS IN TRAIN:", targets)

                optimizer.zero_grad()

                logits, dec_out, Z = multitaskAE(input)  ## model
                # print ("logits train:", logits)

                loss_mse = customLoss()
                loss = loss_mse(input, dec_out, Z, dom_out, logits, targets)

                losses.append(loss)

        final_loss = torch.mean(torch.stack(losses))
        # print("FINAL LOSS:", final_loss)

        final_loss.backward()
        train_loss += final_loss.item()
        optimizer.step()

        if epochs % 1 == 50:
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epochs, train_loss))  # / len(dataloader.dataset)))
            train_losses.append(train_loss)  # / len(dataloader.dataset))


train_AE()
PATH = "/content/gdrive/MyDrive/OOD_generalization/mate-mi-reg-model/meta_latent_model_MTAE_20K.pth"
torch.save(multitaskAE, PATH)
torch.save(multitaskAE.state_dict(), PATH)
print("MODEL SAVED")

# print ("TRAIN LOSS:", train_losses)
print("PLOTTING TRAINING:")
X = np.arange(epochs)
Y = train_losses
plt.plot(X, Y)
plt.savefig('loss_vs_epoch.png')

# # ================= TEST =================
# print('************ TEST ************ \n>>restoring from:' + PATH + '<<')
#
# D_in = 29
# print ("D_in shape:", D_in)
# H = 20
# H2 = 14
#
# model = MultitaskAutoencoder(D_in, H, H2)
# model = model.to(device)
# model.load_state_dict(torch.load(PATH))
# model.eval()
#
# data = pd.read_csv("test_IN9_DDOS_INFIL.csv")
# data[data.select_dtypes(np.float64).columns] = data.select_dtypes(np.float64).astype(np.float32)
# test_data = data.loc[:,data.columns != 'Label']
# test_data = test_data.to_numpy()
#
# test_labels =  data['Label']
# test_labels[test_labels>1.0] = 1.0
# test_labels = test_labels.to_numpy()
#
#
# data = pd.read_csv("test_OOD_RARE.csv")
# data[data.select_dtypes(np.float64).columns] = data.select_dtypes(np.float64).astype(np.float32)
# test_data_ood = data.loc[:,data.columns != 'Label']
# test_data_ood = test_data_ood.to_numpy()
#
# test_labels_ood =  data['Label']
# test_labels_ood[test_labels_ood>1.0] = 1.0
# test_labels_ood = test_labels_ood.to_numpy()
#
# data = pd.read_csv("train_data_final_20_dom_1k_INFIL_DDOS.csv") ### Train data for this model
# data[data.select_dtypes(np.float64).columns] = data.select_dtypes(np.float64).astype(np.float32)
# data_train = data.loc[:,data.columns != 'Label']
#
# train_labels =  data['Label']
# train_labels[train_labels>1.0] = 1.0
# train_labels = train_labels.to_numpy()
#
# encoder_inputs = torch.from_numpy(data_train.to_numpy())
# tr_code = model.encode(encoder_inputs.to(device))
#
#
# test_data = torch.from_numpy(test_data)
# ts_code = model.encode(test_data.to(device))
# #ts_code = ts_code.cpu().detach().numpy()
#
# test_data_ood = torch.from_numpy(test_data_ood)
# ts_code_ood = model.encode(test_data_ood.to(device))
# #
# # dim_red = 1  # perform PCA on the codes and plot the first two components
# # if dim_red:
# #     dim_reduction_plot(ts_code, test_labels, 1)
#
#
# dec_out = model.decode(ts_code)
#
# dec_out = dec_out.cpu().detach().numpy()
# test_data = test_data.cpu().detach().numpy()
#
# recons_loss_test = np.mean((dec_out - test_data) ** 2)
# print("recons_loss_test:", recons_loss_test)
#
# dec_out = np.reshape(dec_out, (test_data.shape[1], test_data.shape[0]))
# dec_out = np.transpose(dec_out, axes=[1, 0])
#
# # MSE and corr
# #test_mse, test_corr, tot_mae = mse_and_corr(test_data, test_labels[:], dec_out, test_len)
# #print('Test MSE: %.5f\nTest Pearson correlation: %.3f\nTest MAE: %.5f' % (test_mse, test_corr, tot_mae))
#
#
# print("Test labels:", test_labels.shape)
# acc, prec, recall ,f1, auc, acc_ood, prec_ood, recall_ood, auc_ood = classify_with_knn(tr_code,train_labels,ts_code, test_labels[:], ts_code_ood, test_labels_ood, k =3)
# print('kNN -- acc: %.3f, prec: %.3f, recall: %.3f, F1: %.3f, AUC: %.3f, ACC_OOD: %.3f, PREC_OOD: %.3f, RECALL_OOD: %.3f, AUC_OOD: %.3f' % (acc, prec,recall,f1, auc, acc_ood, prec_ood, recall_ood,auc_ood))




