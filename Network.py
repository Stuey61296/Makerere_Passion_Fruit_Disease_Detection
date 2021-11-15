# imports
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint

from Utils import log, convert_to_array


class Network:

    def __init__(self, _model, _name):
        self.compiled = False
        self.model = _model
        self.name = _name
        self.history = {"Train": pd.DataFrame(), "Test": pd.DataFrame()}

    def compile(self, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', 'mse'])
        self.compiled = True

    def train(self, train_x, train_y, validation_x, validation_y, epoch, batch_size, learning_rate):
        log("info", f"Training {self.name} on dataset")
        if not self.compiled:
            self.compile(learning_rate=learning_rate)

        train_x = convert_to_array(train_x, "Train")
        train_y = np.array(train_y)
        validation_x = convert_to_array(validation_x, "Validation")
        validation_y = np.array(validation_y)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=epoch // 10, verbose=0, mode='auto')
        history = self.model.fit(train_x, train_y, epochs=epoch, validation_data=(validation_x, validation_y),
                                 verbose=1, batch_size=batch_size, callbacks=[early_stop])
        self.history["Train"] = pd.DataFrame(history.history)
        log("info", "Training Done")
        return self

    def test(self, data, labels, batch_size):
        log("info", f"Testing {self.name}")
        loss, mse, acc = self.model.evaluate(data, labels, verbose=1, batch_size=batch_size)
        rmse = math.sqrt(mse)
        return rmse

    def get_history(self):
        return self.history
