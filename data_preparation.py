import numpy as np
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler


def prepare_data(data, seq_length, train_set_ratio):
    """
    Prepare dataset (array) to normalized train-test set (tensor) for model training

    Args:
        data (np.array): dataset
        seq_length (int): sequence lenght of input data
        train_set_ratio (float)

    Returns:
        data_x, data_y, train_x, train_y, test_x, test_y (tensor)
        train_size (int)
        scaler (sklearn.scaler)
    """
    scaled_data, scaler = data_normalization(data)

    x, y = sliding_windows(scaled_data, seq_length)

    data_x = Variable(torch.Tensor(np.array(x)))
    data_y = Variable(torch.Tensor(np.array(y)))

    train_size = int(train_set_ratio * len(y))
    train_x, train_y, test_x, test_y = train_test_split(x, y, train_size)
    return data_x, data_y, train_x, train_y, test_x, test_y, train_size, scaler


def train_test_split(x, y, train_size):
    """
    Split data to train and test set

    Args:
        x (np.array)
        y (np.array)
        train_size (int)

    Returns:
        train_x, train_y, test_x, test_y (tensor)
    """
    train_x = Variable(torch.Tensor(np.array(x[0:train_size])))
    train_y = Variable(torch.Tensor(np.array(y[0:train_size])))

    test_x = Variable(torch.Tensor(np.array(x[train_size : len(x)])))
    test_y = Variable(torch.Tensor(np.array(y[train_size : len(y)])))
    return train_x, train_y, test_x, test_y


def data_normalization(data):
    """
    Normalize data wit min-max scaler

    Args:
        data (np.array)

    Returns:
        scaled_data (np.array)
        scaler (sklearn.scaler)
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def sliding_windows(data, seq_length):
    """
    Prepare data to time series format with seq_length x input and 1 lenght y output

    Args:
        data (np.array)
        seq_length (int)

    Returns:
        x, y (np.array)
    """
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i : (i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)
