import numpy as np
from sklearn.metrics import classification_report
import scipy.io as sio

target_name5 = ["类别"+str(i) for i in range(5)]
target_name10 = ["类别"+str(i) for i in range(10)]
target_name15 = ["类别"+str(i) for i in range(15)]
target_name20 = ["类别"+str(i) for i in range(20)]
target_name25 = ["类别"+str(i) for i in range(25)]
target_name30 = ["类别"+str(i) for i in range(30)]


def read_data():

    with open("result/10/lab_ts.txt", "r") as f1:
        lab_index = f1.read().splitlines()
    with open("result/10/pre_ts.txt", "r") as f1:
        pre_index = f1.read().splitlines()

    readata = sio.loadmat("data/api/10/Apis_811_des.mat")
    index = readata['test_idx'].tolist()

    labels = []
    predict = []
    for i in index[0]:
        labels.append(lab_index[i])
        predict.append(pre_index[i])

    print(classification_report(labels, predict, target_names=target_name10, digits=4))


if __name__ == "__main__":
    read_data()