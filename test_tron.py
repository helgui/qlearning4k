from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Input, Flatten
from keras.optimizers import Adam, RMSprop

from qlearning4k.games import Tron

model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(20, 30, 4)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='linear'))

model.compile(optimizer=Adam(), loss='mse')


