import numpy as np
import pickle
from batch_and_increment import rho
import os


def pocket_update(x, y, w):
    return y * x


def pocket_condition(a, b):
    return a <= b


def perceptron(w_pocket, sota, data, epoch, update_function, update_condition, rho):
    w = np.array(w_pocket)
    l_rate = rho(epoch)
    for X, y in zip(data[0], data[1]):
        if y * np.dot(X, w) < 0:
            w += l_rate * update_function(X, y, w)
    mispredictions = np.sum(np.less_equal(np.multiply(data[1], np.dot(data[0], w)), np.zeros(data[1].shape)))
    if update_condition(mispredictions, sota):
        return w, mispredictions
    return w_pocket, sota


def train_perceptron(data, epochs, filename, seed=None, rho=rho, update_function=pocket_update, update_condition=pocket_condition):
    np.random.seed(seed)
    w_pocket = np.random.rand(data[0].shape[1])*0.02 - 0.01
    w_pocket[-1] = 0
    sota = len(data[0])
    sota_series = []
    for epoch in range(epochs):
        w_pocket, sota = perceptron(w_pocket, sota, data, epoch, update_function, update_condition, rho)
        sota_series.append(sota)
    w = np.array(w_pocket)
    np.savetxt(filename, w, delimiter=",")
    return sota_series


if __name__ == '__main__':
    for d in range(1, 4):
        for m in [5, 10, 20]:
            print(d, m)
            with open("data_"+str(d)+"/data"+str(m)+".txt", "rb") as f:
                data = pickle.load(f)
            print(train_perceptron(data, 300, "data_"+str(d)+"/pocket_"+str(m)+".csv")[-1])
