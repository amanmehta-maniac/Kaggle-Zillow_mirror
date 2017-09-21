from __future__ import print_function
import keras
from keras.model import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras import backend
from parse_data import parse_data()

X_train, Y_train, X_test, Y_test = parse_data()
