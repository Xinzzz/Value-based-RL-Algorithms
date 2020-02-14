import os
from typing import Dict, List, Tuple
import pickle
import numpy as np

def save_model(model, data):
    path = 'logs/' + model + '.pickle'
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(model, color):
    path = 'logs/' + model + '.pickle'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    store = data[0], data[1], data[2], data[3], model, color
    return store

def load_model(model):
    path = 'logs/' + model + '.pickle'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data