from lib import model_GATLSTM_period
import numpy as np
import matplotlib.pyplot as plt
import time
from pandas import read_csv
# import scipy.sparse
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import math

import torch
import torch.utils.data as Data



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
    usegpu = True
    # ********************************************************************************
    # roads
    df_roads = read_csv('data/road/adjacent_matrix_undirected_156.csv', header=0, index_col=0)
    road_list = df_roads.columns.values.tolist()
    print("there are " + str(len(road_list)) + " roads")
    values_A = df_roads.values.astype(np.float32)
    # Feature graph
    # A = scipy.sparse.csr_matrix(values_A)
    A = torch.FloatTensor(values_A)
    print("A: ", A.shape)

    # speed data
    df_speed = read_csv('data/speed/15min_0108.csv', header=0, index_col=0)
    df_speed = df_speed[road_list]
    values = df_speed.values
    print("the size of speed data: ", values.shape)

    # embedding
    embedding = read_csv('data/embedding/road2vec_embedding_50.csv', header=0)
    embedding = embedding[road_list]
    embedding = embedding.values
    print("the size of embedding data: ", embedding.shape)

    # define parameters
    n_time = int(values.shape[0])
    seq_length = 4
    period_steps = 7  # 7 days
    n_samples = n_time - seq_length - int(period_steps * 24 * (60 / 15))
    n_test = int(6 * 24 * (60 / 15))  # the last 6 days
    n_train = int((n_samples - n_test) * 4 / 5)
    n_val = int((n_samples - n_test) - n_train)
    print("=======================================================")
    print("n_time: ", n_time)
    print("seq_length: ", seq_length, "time intervals")
    print("period_steps: ", period_steps, "days")
    print("n_samples: ", n_samples)
    print("n_train: ", n_train)
    print("n_val: ", n_val)
    print("n_test: ", n_test)
    print("=======================================================")
    values_test = values[-n_test:, :]
    print("the size of test data: ", values_test.shape)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    # values_low = []
    # 需要的是values 它的size是(2880, 156)
    values_scaler = scaler.fit_transform(values)
    # values_low_scaler = scaler.fit_transform(values_low)
    dataX = []
    dataY = []
    start = int(period_steps * (60 / 15 * 24))
    end = n_time - seq_length
    for i in range(start, end, 1):
        input_data = values_scaler[i:i + seq_length]
        input_data = [k for k in input_data]
        for j in range(period_steps):
            t = i - int(j * (60 / 15 * 24))
            input_data.append(values_scaler[t])
        output = values_scaler[i + seq_length]
        dataX.append(input_data)
        dataY.append(output)

    # reshape input to be ndarray/csr_matrix
    X = np.reshape(dataX, (len(dataX), seq_length + period_steps, values_scaler.shape[1])).astype(np.float32)
    y = np.reshape(dataY, (len(dataY), values_scaler.shape[1])).astype(np.float32)

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
    X_val = torch.Tensor(X_val)
    y_val = torch.FloatTensor(y_val)
    embedding = torch.FloatTensor(embedding)
    embedding = torch.transpose(embedding, dim0=0, dim1=1)

    # Batch loader
    traindataset = Data.TensorDataset(X_train, y_train)
    trainloader = Data.DataLoader(
        dataset=traindataset,  # torch TensorDataset format
        batch_size=50,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=2,  # subprocesses for loading data
    )
    model = model_GATLSTM_period.GAT_LSTM(nfeat=50, nhid=32, dropout=0.5, alpha=0.2)
    print(model)
    # optimizer = torch.optim.Adam(model.parameters(),
    #                        lr=0.01, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-5, momentum=0.9, nesterov=True)
    criterion = torch.nn.MSELoss()

    device = torch.device("cuda:0")
    if usegpu:
        model.to(device)
        X_test.to(device)
        X_val.to(device)
        y_val.to(device)
        embedding.to(device)
        A.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-5, momentum=0.9, nesterov=True)
        criterion.to(device)
    X_val.to(device)
    y_val.to(device)

    # Training
    t_total = time.time()
    losstrain = []
    lossval = []
    for epoch in range(300):
        t = time.time()
        for i, data in enumerate(trainloader):
            traindata, truevalues = data
            if usegpu:
                traindata, truevalues = traindata.to(device), truevalues.to(device)
            optimizer.zero_grad()
            out = model(embedding.cuda(), traindata, A.cuda())
            loss = criterion(out, truevalues)
            loss.backward()
            optimizer.step()
            val_out = model(embedding.cuda(), X_val.cuda(), A.cuda())
            loss_val = criterion(val_out, y_val.cuda())
        losstrain.append(loss.item())
        lossval.append(loss_val.item())
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'time for one epoch: {:.4f}s'.format(time.time() - t))

    print("Training Finished!")
    # print("Optimization Finished!")
    torch.save(model.state_dict(), '{}.pkl'.format('./pre-trained-models/model_GATLSTM_period_params'))
    print("Total training time elapsed: {:.4f}s".format(time.time() - t_total))

    # plot the loss
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(losstrain, 'b.-')
    ax1.set_ylabel('training loss', color='b')
    ax2 = ax1.twinx()
    ax2.plot(lossval, 'g.-')
    ax2.set_ylabel('validation loss', color='g')
    plt.show()

    # Test
    if usegpu:
        y_pre_test = model(embedding.cuda(), X_test.cuda(), A.cuda())
        y_pre_test = y_pre_test.cpu().detach().numpy()
        y_pre_test = scaler.inverse_transform(y_pre_test)
        y_true_test = scaler.inverse_transform(y_test)
    else:
        y_pre_test = model(embedding, X_test, A)
        y_pre_test = y_pre_test.detach().numpy()
        y_pre_test = scaler.inverse_transform(y_pre_test)
        y_true_test = scaler.inverse_transform(y_test)

    print("the size of the prediction: ", y_pre_test.shape)
    [mse, rmse, mae, mape] = evaluate_result(y_pre_test, values_test)
    plot_prediction(y_pre_test, values_test)
