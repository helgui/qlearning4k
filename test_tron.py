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
model.add(Conv2D(16, (3, 3), data_format=DF, padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), data_format=DF, padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), data_format=DF, padding='same', activation='relu'))
model.add(MaxPool2D((2, 2), data_format=DF))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='linear'))

model.compile(optimizer=Adam(0.02), loss='mse')

enemy_model = Sequential.from_config(model.get_config())
enemy_model.compile(optimizer=Adam(0.02), loss='mse')

tron = Tron(enemy_model)
agent = Agent(model=model)

'''
print(tron.pos)
print(tron.active)
cv2.imshow('tron', tron.draw_img())
cv2.waitKey()

while not tron.is_over():
	pa = tron.get_possible_actions()
	tron.play(pa[0])
	print('-------')
	print(tron.pos)
	print(tron.active)
	cv2.imshow('tron', tron.draw_img())
	cv2.waitKey()
'''
agent.train(game=tron, epsilon=(1.0, 0.05), batch_size=32, nb_epoch=10000, gamma=0.9)
agent.play(tron, nb_epoch=1)
