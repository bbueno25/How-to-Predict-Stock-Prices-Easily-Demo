"""
DOCSTRING
"""
import keras
import matplotlib.pyplot as pyplot
import numpy
import time
import warnings

warnings.filterwarnings('ignore')

def build_model(layers):
    """
    DOCSTRING
    """
    model = models.Sequential()
    model.add(keras.layers.recurrent.LSTM(input_dim=layers[0], units=layers[1], return_sequences=True))
    model.add(keras.layers.core.Dropout(0.2))
    model.add(keras.layers.recurrent.LSTM(layers[2], return_sequences=False))
    model.add(keras.layers.core.Dropout(0.2))
    model.add(keras.layers.core.Dense(units=layers[3]))
    model.add(keras.layers.core.Activation('linear'))
    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print('Compilation Time:{}'.format(time.time() - start))
    return model

def load_data(filename, seq_len, normalize_window):
    """
    DOCSTRING
    """
    f = open(filename, 'r').read()
    data = f.split('\n')
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    if normalize_window:
        result = normalize_windows(result)
    result = numpy.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    numpy.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    x_train = numpy.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = numpy.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_train, y_train, x_test, y_test

def normalize_windows(window_data):
    """
    DOCSTRING
    """
    normalized_data = []
    for window in window_data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return normalized_data

def plot_results_multiple(predicted_data, true_data, prediction_len):
    """
    DOCSTRING
    """
    fig = pyplot.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        pyplot.plot(padding + data, label='Prediction')
        pyplot.legend()
    pyplot.show()

def predict_point_by_point(model, data):
    """
    Predict each timestep given the last sequence of true data,
    in effect only predicting 1 step ahead each time.
    """
    predicted = model.predict(data)
    return numpy.reshape(predicted, (predicted.size,))

def predict_sequence_full(model, data, window_size):
    """
    Shift the window by 1 new prediction each time,
    re-run predictions on new window
    """
    curr_frame = data[0]
    predicted = []
    for _ in range(len(data)):
        predicted.append(model.predict(curr_frame[numpy.newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = numpy.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    """
    Predict sequence of 50 steps before shifting prediction run forward by 50 steps.
    """
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for _ in range(prediction_len):
            predicted.append(model.predict(curr_frame[numpy.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = numpy.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

if __name__ == '__main__':
    # Step 1. Load Data
    X_train, y_train, X_test, y_test = load_data('sp500.csv', 50, True)
    # Step 2. Build Model
    model = keras.models.Sequential()
    # 1st Layer
    model.add(keras.layers.recurrent.LSTM(input_dim=1, units=50, return_sequences=True))
    model.add(keras.layers.core.Dropout(0.2))
    # 2nd Layer
    model.add(keras.layers.recurrent.LSTM(100, return_sequences=False))
    model.add(keras.layers.core.Dropout(0.2))
    # 3rd Layer
    model.add(keras.layers.core.Dense(units=1))
    model.add(keras.layers.core.Activation('linear'))
    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print('compilation time:{}'.format(time.time() - start))
    # Step 3. Train the model
    model.fit(X_train, y_train, batch_size=512, epochs=1, validation_split=0.05)
    # Step 4. Plot the predictions
    predictions = predict_sequences_multiple(model, X_test, 50, 50)
    plot_results_multiple(predictions, y_test, 50)
