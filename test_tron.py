from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Input, Flatten, Reshape
from keras.optimizers import Adam, RMSprop
import numpy as np
from matplotlib import pyplot as plt
from qlearning4k.games import Tron
from qlearning4k import Agent
import cv2

DF = 'channels_first'

model = Sequential()
model.add(Reshape((4, 20, 30), input_shape=(1, 4, 20, 30)))
model.add(Conv2D(16, (5, 5), data_format=DF, padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), data_format=DF, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='linear'))

#print(model.summary())

model.compile(optimizer=Adam(1e-2), loss='mse')

enemy_model = Sequential.from_config(model.get_config())
enemy_model.compile(optimizer=Adam(1e-2), loss='mse')

tron = Tron(enemy_model)
agent = Agent(model=model)

print('Initial phase')

agent.train(game=tron, epsilon=(1.0, 0.1), epsilon_rate=0.5, batch_size=32, nb_epoch=10000, gamma=0.9, checkpoint=500)
agent.play(tron, nb_epoch=1)
tron.update_ai_model(agent.model)

for i in range(10):
	print('Phase #', i + 1)
	agent.train(game=tron, epsilon=0.1, batch_size=32, nb_epoch=10000, gamma=0.9, checkpoint=100)
	agent.play(tron, nb_epoch=1)
	game.update_ai_model()