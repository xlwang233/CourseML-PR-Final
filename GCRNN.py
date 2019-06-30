from lib import model_gcnlstm
import numpy as np
import matplotlib.pyplot as plt
import time
from pandas import read_csv
import scipy.sparse
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from utils import normalize



# ********************************************************************************
# function of evaluate
def evaluate_result(y_pre, y_true):
    print("------------------------------")
    mse = ((y_pre - y_true) ** 2).mean()
    print("mse = ", mse)
    rmse = math.sqrt(((y_pre - y_true) ** 2).mean())
    print("rmse = ", rmse)
    mae = abs(y_pre - y_true).mean()
    print("mae = ", mae)
    mape = (abs(y_pre - y_true) / abs(y_true)).mean() * 100
    print("mape = ", mape)
    print("------------------------------")
    return mse, rmse, mae, mape


# function of plot prediction
def plot_prediction(y_pre, y_true):
    plot_road = [0, 10, 50, 100, 150]
    for i in plot_road:
        plt.figure(figsize=(20, 10))
        plt.plot(range(y_true.shape[0]), y_true[:, i], label='TrueValue')
        plt.plot(range(y_pre.shape[0]), y_pre[:, i], label='PredictedValue')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Time step', fontsize=18)
        plt.ylabel('km/h', fontsize=18)
        plt.legend()
        plt.title('Road_' + str(i), fontsize=20, color='red')
        plt.show()


if __name__ == "__main__":
    usegpu = False
    # ********************************************************************************
    # roads
    df_roads = read_csv('data/road/adjacent_matrix_undirected_156.csv', header=0, index_col=0)
    road_list = df_roads.columns.values.tolist()
    print("there are " + str(len(road_list)) + " roads")
    values_A = df_roads.values.astype(np.float32)
    # Feature graph
    # A = scipy.sparse.csr_matrix(values_A)
    values_A = normalize(values_A + np.eye(values_A.shape[0]))
    A = torch.FloatTensor(values_A)
    print(A[:20, :20])
    print("A: ", A.shape)


    # speed data
    df_speed = read_csv('data/speed/15min_0108.csv', header=0, index_col=0)
    df_speed = df_speed[road_list]
    values = df_speed.values
    print("the size of speed data: ", values.shape)

    # define parameters
    n_time = int(values.shape[0])
    seq_length = 4
    n_samples = n_time - seq_length
    n_test = int(6 * 24 * (60 / 15))  # the last 6 days
    n_train = int((n_samples - n_test) * 4 / 5)
    n_val = int((n_samples - n_test) - n_train)
    print("=======================================================")
    print("n_time: ", n_time)
    print("seq_length: ", seq_length)
    print("n_samples: ", n_samples)
    print("n_train: ", n_train)
    print("n_val: ", n_val)
    print("n_test: ", n_test)
    print("=======================================================")
    values_test = values[-n_test:, :]
    print("the size of test data: ", values_test.shape)

    # predict speed
    print("============================================================= predict speed of test")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    values_scaler = scaler.fit_transform(values)

    dataX = []
    dataY = []
    for i in range(0, len(values_scaler) - seq_length, 1):
        input = values_scaler[i:i + seq_length]
        output = values_scaler[i + seq_length]
        dataX.append([j for j in input])
        dataY.append(output)

    # reshape input to be ndarray/csr_matrix
    X = np.reshape(dataX, (len(dataX), seq_length, values.shape[1])).astype(np.float32)
    y = np.reshape(dataY, (len(dataY), values.shape[1])).astype(np.float32)

    X_train = X[:n_train]
    X_val = X[n_train:-n_test]
    X_test = X[-n_test:]

    y_train = y[:n_train]
    y_val = y[n_train:-n_test]
    y_test = y[-n_test:]

    print("------------------------------")
    print("the shape of X: ", X.shape)
    print("the shape of y: ", y.shape)
    print("the shape of X_train: ", X_train.shape)
    print("the shape of X_val: ", X_val.shape)
    print("the shape of X_test: ", X_test.shape)
    print("the shape of y_train: ", y_train.shape)
    print("the shape of y_val: ", y_val.shape)
    print("the shape of y_test: ", y_test.shape)
    print("------------------------------")

    del dataX
    del dataY


    # Convert data from numpy array to pytorch tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    # y_test = torch.FloatTensor(y_test)

    # Batch loader
    traindataset = Data.TensorDataset(X_train, y_train)
    trainloader = Data.DataLoader(
        dataset=traindataset,  # torch TensorDataset format
        batch_size=50,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=2,  # subprocesses for loading data
    )
    model = model_gcnlstm.GCN_LSTM(noutput=156, dropout=0.5)
    print(model)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                        lr=0.01, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5, momentum=0.9, nesterov=True)
    criterion = torch.nn.MSELoss()

    if usegpu:
        device = torch.device("cuda:0")
        model.to(device)
        X_test.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5, momentum=0.9, nesterov=True)
        criterion.to(device)

    # Training
    t_total = time.time()
    losslist = []
    for epoch in range(200):
        t = time.time()
        for i, data in enumerate(trainloader):
            traindata, truevalues = data
            # traindata, truevalues = traindata.to(device), truevalues.to(device)
            optimizer.zero_grad()
            out = model(traindata, A)
            loss = criterion(out, truevalues)
            loss.backward()
            optimizer.step()
        losslist.append(loss.item())
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'time for one epoch: {:.4f}s'.format(time.time() - t))

    print("Training Finished!")
    # print("Optimization Finished!")
    torch.save(model.state_dict(), '{}.pkl'.format('./models/model_params'))
    print("Total training time elapsed: {:.4f}s".format(time.time() - t_total))

    # plot the loss
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(losslist, 'b.-')
    ax1.set_ylabel('training loss', color='b')
    plt.show()

    # Test
    # X_test.to(device)
    # print(X_test.device)

    if usegpu == True:
        y_pre_test = model(X_test.cuda())
        y_pre_test = y_pre_test.cpu().detach().numpy()
        y_pre_test = scaler.inverse_transform(y_pre_test)
        y_true_test = scaler.inverse_transform(y_test)
    else:
        y_pre_test = model(X_test, A)
        y_pre_test = y_pre_test.detach().numpy()
        y_pre_test = scaler.inverse_transform(y_pre_test)
        y_true_test = scaler.inverse_transform(y_test)

    print("the size of the prediction: ", y_pre_test.shape)
    [mse, rmse, mae, mape] = evaluate_result(y_pre_test, values_test)
    plot_prediction(y_pre_test, values_test)
