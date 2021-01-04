import scipy.io as sio
import numpy as np


impath='data/imdb/imdb5k.mat'  # MAM  MDM  MYM
acmpath='data/acm/acm3025.mat'  # PTP  PLP  PAP
apipath='data/api/Apis.mat'  # PTP  PLP  PAP
data = sio.loadmat(impath)
print(data["MAM"].shape)
print(data["MDM"].shape)
print(data["MYM"].shape)
print(data["feature"].shape)
print(data["label"].shape)
# print(data["MDM"].shape)
# print(data["feature"].shape)
print(data["train_idx"].shape)
print(data["val_idx"].shape)
print(data["test_idx"].shape)
# truelabels, truefeatures = data['label'], data['feature'].astype(float)
# #MAM, MDM, MYM = data['MAM'], data['MDM'], data['MYM'].astype(float)
# PTP, PLP, PAP = data['PTP'], data['PLP'], data['PAP'].astype(float)
# dataNew = 'dataNew.mat'
# m1 = np.array([[0,0,0],[0,1,0]])
# m2 = np.array([[1,0,0],[0,1,0]])
# data = {"amama":m1, "ata":m2}
# sio.savemat(dataNew, data)
# da = sio.loadmat(dataNew)
# print(da)