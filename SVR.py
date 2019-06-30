import numpy as np
from pandas import read_csv, read_excel
import sklearn
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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


# ********************************************************************************
# roads
df_roads = read_csv('data/road/adjacent_matrix_undirected_156.csv', header=0, index_col=0)
road_list = df_roads.columns.values.tolist()
print("there are " + str(len(road_list)) + " roads")

# speed data
df_speed = read_csv('data/speed/15min.csv', header=0, index_col=0)
df_speed = df_speed[road_list]
values = df_speed.values
print("the size of speed data: ", values.shape)

# define parameters
n_time = int(values.shape[0])
seq_length = 4
n_samples = n_time - seq_length
n_test = int(6 * 24 * (60 / 15))  # the last 6 days
print("=======================================================")
print("n_time: ", n_time)
print("seq_length: ", seq_length)
print("n_samples: ", n_samples)
print("n_test: ", n_test)
print("=======================================================")
values_test = values[-n_test:, :]
print("the size of test data: ", values_test.shape)

# predict speed
print("============================================================= predict speed of test")

clf = SVR(C=1.0, cache_size=200, coef0=1.0, degree=3, epsilon=0.001, gamma='auto',
          kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

scaler = MinMaxScaler(feature_range=(0, 1))
values_scaler = scaler.fit_transform(values)

y_pre = []

for j in range(len(road_list)):
    values_road = values_scaler[:, j]
    #  prepare the input and output
    dataX = []
    dataY = []
    for i in range(0, len(values_road) - seq_length, 1):
        data_input = values_road[i:i + seq_length]
        data_output = values_road[i + seq_length]
        dataX.append([j for j in data_input])
        dataY.append(data_output)

    # reshape input to be ndarray/csr_matrix
    X = np.reshape(dataX, (len(dataX), seq_length))
    y = np.reshape(dataY, (len(dataY),))

    X_train = X[:-n_test]
    X_test = X[-n_test:]
    y_train = y[:-n_test]
    y_test = y[-n_test:]

    del dataX
    del dataY

    clf.fit(X_train, y_train)
    y_pre_test = clf.predict(X_test)
    y_pre.append(y_pre_test)

print("------------------------------")
print("the shape of X_train: ", X_train.shape)
print("the shape of X_test: ", X_test.shape)
print("the shape of y_train: ", y_train.shape)
print("the shape of y_test: ", y_test.shape)
print("------------------------------")
y_pre = np.array(y_pre).transpose()
y_pre = scaler.inverse_transform(y_pre)
print("the size of the prediction: ", y_pre.shape)
[mse, rmse, mae, mape] = evaluate_result(y_pre, values_test)
plot_prediction(y_pre, values_test)