import numpy as np
import pickle

a = 1

for m in [5, 10, 20]:
    data = [np.zeros((1000,50)), np.zeros(1000)]
    for i in range(1000):
        if np.random.rand() < 0.5:
            data[0][i,:m] = (np.random.rand(m) * a) - 2*a
            data[1][i] = -1
        else:
            data[0][i,:m] = (np.random.rand(m) * a) + a
            data[1][i] = 1
    data[0][:,m:] = (np.random.rand(1000, 50-m) * 4*a) - 2*a
    with open("data"+str(m)+".txt", "wb") as f:
        pickle.dump(data, f)
