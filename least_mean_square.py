import numpy as np
import pickle
from pocket import train_perceptron
import matplotlib.pyplot as plt


def lms_update(x, y, w):
    return (y - np.dot(x, w)) * x


def lms_condition(a, b):
    return True


if __name__ == "__main__":
    for d in range(1, 4):
        for m in [5, 10, 20]:
            with open("data_"+str(d)+"/data"+str(m)+".txt", "rb") as f:
                data = pickle.load(f)
            print(d, m)
            print(train_perceptron(data, 200, "data_"+str(d)+"/least_mean_square_"+str(m)+"_1.csv", update_function=lms_update, update_condition=lms_condition)[-1])
