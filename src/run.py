from keras.layers import core
from keras.layers import recurrent
from keras import models
from predict_stock_prices import lstm
from time import time

if __name__ == '__main__':
    # Step 1. Load Data
    X_train, y_train, X_test, y_test = lstm.load_data('data/sp500.csv', 50, True)
    # Step 2. Build Model
    model = models.Sequential()
    # 1st Layer
    model.add(recurrent.LSTM(input_dim=1, units=50, return_sequences=True))
    model.add(core.Dropout(0.2))
    # 2nd Layer
    model.add(recurrent.LSTM(100, return_sequences=False))
    model.add(core.Dropout(0.2))
    # 3rd Layer
    model.add(core.Dense(units=1))
    model.add(core.Activation('linear'))
    start = time()
    model.compile(loss='mse', optimizer='rmsprop')
    print('compilation time:{}'.format(time() - start))
    # Step 3. Train the model
    model.fit(X_train, y_train, batch_size=512, nb_epoch=1, validation_split=0.05)
    # Step 4. Plot the predictions
    predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
    lstm.plot_results_multiple(predictions, y_test, 50)
