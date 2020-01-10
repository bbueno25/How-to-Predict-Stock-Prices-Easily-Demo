from keras import models
from keras.layers import core
from keras.layers import recurrent
from matplotlib import pyplot as plt
from numpy import array as np_array
from numpy import insert as np_insert
from numpy import newaxis
from numpy import random as np_random
from numpy import reshape as np_reshape
from os import environ as os_environ
from time import time
from warnings import filterwarnings

filterwarnings('ignore')

def build_model(layers):
    model = models.Sequential()
    model.add(recurrent.LSTM(input_dim=layers[0], units=layers[1], return_sequences=True))
    model.add(core.Dropout(0.2))
    model.add(recurrent.LSTM(layers[2], return_sequences=False))
    model.add(core.Dropout(0.2))
    model.add(core.Dense(units=layers[3]))
    model.add(core.Activation('linear'))
    start = time()
    model.compile(loss='mse', optimizer='rmsprop')
    print('Compilation Time:{}'.format(time() - start))
    return model

def load_data(filename, seq_len, normalize_window):
    f = open(filename, 'r').read()
    data = f.split('\n')
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    if normalize_window:
        result = normalize_windows(result)
    result = np_array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np_random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    x_train = np_reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np_reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_train, y_train, x_test, y_test

def normalize_windows(window_data):
    normalized_data = []
    for window in window_data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return normalized_data

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def predict_point_by_point(model, data):
    """ Predict each timestep given the last sequence of true data,
        in effect only predicting 1 step ahead each time """
    predicted = model.predict(data)
    return np_reshape(predicted, (predicted.size,))

def predict_sequence_full(model, data, window_size):
    """ Shift the window by 1 new prediction each time,
        re-run predictions on new window """
    curr_frame = data[0]
    predicted = []
    for _ in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np_insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    """ Predict sequence of 50 steps before shifting prediction run forward by 50 steps """
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for _ in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np_insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs
