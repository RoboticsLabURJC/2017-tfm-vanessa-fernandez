#  Based on: https://arxiv.org/pdf/1604.07316.pdf
#
#  Authors :
#       Vanessa Fernandez Martinez <vanessa_1895@msn.com>

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Dropout, ConvLSTM2D, Reshape, Lambda, MaxPooling2D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.regularizers import l2


def pilotnet_model(img_shape):
    '''
    Model of End to End Learning for Self-Driving Cars (NVIDIA)
    '''
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    adam = Adam(lr=0.00001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model


def tinypilotnet_model(img_shape):
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model


def lstm_tinypilotnet_model(img_shape):
    model = Sequential()
    #model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), input_shape=img_shape, activation="relu"))
    model.add(Conv2D(16, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(32, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Reshape((1, 14, 19, 32)))
    model.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3, border_mode='same', return_sequences=True))
    model.add(Reshape((14, 19, 40)))
    model.add(Conv2D(1, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Flatten())
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model


def lstm_model(img_shape):
    # https://github.com/udacity/self-driving-car/blob/master/steering-models/community-models/chauffeur/models.py
    # https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html
    # https://github.com/BoltzmannBrain/self-driving

    model = Sequential()
    #model.add(LSTM(units = 50, return_sequences = True, input_shape = img_shape))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units = 50, return_sequences = True))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=50, return_sequences=True))
    #model.add(Dropout(0.2))
    #model.add(LSTM(units=50))
    #model.add(Dropout(0.2))
    #model.add(Dense(units=1))

    # img_shape = (17341, img_shape[0], img_shape[1], img_shape[2])
    #
    #img_shape = (17341, 10, img_shape[0], img_shape[1], img_shape[2])
    #####model.add(Reshape((17341, 10, img_shape[0], img_shape[1], img_shape[2])))
    # model.add(TimeDistributed(Conv2D(2, (1, 1), padding='same', activation='elu', kernel_regularizer='l2'),
    #                           input_shape=img_shape))
    # model.add(TimeDistributed(Conv2D(8, (3, 3), padding='same', activation='elu', kernel_regularizer='l2')))
    # model.add(TimeDistributed(MaxPooling2D(padding='same')))
    # model.add(TimeDistributed(Conv2D(4, (3, 3), padding='same', activation='elu', kernel_regularizer='l2')))
    # model.add(TimeDistributed(MaxPooling2D(padding='same')))
    # model.add(TimeDistributed(Conv2D(2, (3, 3), padding='same', activation='elu', kernel_regularizer='l2')))
    # model.add(TimeDistributed(MaxPooling2D(padding='same')))
    # model.add(TimeDistributed(Conv2D(1, (3, 3), padding='same', activation='elu', kernel_regularizer='l2')))
    # model.add(TimeDistributed(MaxPooling2D(padding='same')))
    #
    # model.add(TimeDistributed(Flatten()))
    # model.add(GRU(12))
    # model.add(Dense(32, activation='elu', kernel_regularizer='l2'))
    # model.add(Dense(1, activation=None))
    # adam = Adam(lr=0.0001)
    # model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])

    #img_shape = (None, 17341, img_shape[0], img_shape[1], img_shape[2])
    #model.add(Reshape((17341, 10, img_shape[0], img_shape[1], img_shape[2])))
    # from keras.models import Model
    # from keras.layers import Input
    # x_input = Input(shape=(10, img_shape[0], img_shape[1], img_shape[2]))
    # x_output = Conv2D(24, (5, 5), init="he_normal", activation='relu', subsample=(5, 4),
    #                                         border_mode='valid')(x_input)
    # base_model = Model(x_input, x_output)
    # model.add(TimeDistributed(base_model, input_shape=base_model.input_shape))

    #model.add(Lambda(
    #    lambda x: x / 127.5 - 1.,
    #    batch_input_shape=(17341, 10, img_shape[0], img_shape[1], img_shape[2]),
    #))
    model.add(TimeDistributed(Conv2D(24, (5, 5), init="he_normal", activation='relu', subsample=(5, 4),
                                     border_mode='valid'), input_shape=img_shape))
    model.add(TimeDistributed(Conv2D(32, (5, 5), init="he_normal", activation='relu', subsample=(3, 2),
                                     border_mode='valid')))
    model.add(TimeDistributed(Conv2D(48, (3, 3), init="he_normal", activation='relu', subsample=(1, 2),
                                     border_mode='valid')))
    model.add(TimeDistributed(Conv2D(64, (3, 3), init="he_normal", activation='relu', border_mode='valid')))
    model.add(TimeDistributed(Conv2D(128, (3, 3), init="he_normal", activation='relu', subsample=(1, 2),
                                     border_mode='valid')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
    model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
    model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=256, init='he_normal', activation='relu', W_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=1, init='he_normal', W_regularizer=l2(0.001)))
    model.compile(loss="mse", optimizer='adadelta', metrics=['accuracy', 'mse', 'mae'])
    return model