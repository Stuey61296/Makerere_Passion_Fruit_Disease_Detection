# imports
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, MaxPooling2D, ZeroPadding2D, Input, Dropout

# inherit
from Network import Network


class CNN_Network(Network):
    def __init__(self, input_shape):
        self.activation_func = 'relu'
        model = Sequential()
        model.add(ZeroPadding2D(padding=(5, 5), input_shape=input_shape))

        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation=self.activation_func))

        model.add(MaxPool2D(strides=2))

        model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation=self.activation_func))
        model.add(MaxPool2D(strides=2))

        model.add(Flatten())

        model.add(Dense(132, activation=self.activation_func))
        model.add(Dropout(0.2))

        model.add(Dense(82, activation=self.activation_func))
        model.add(Dropout(0.2))

        model.add(Dense(3, activation='softmax'))

        super().__init__(_model=model, _name="CNN")
        return
