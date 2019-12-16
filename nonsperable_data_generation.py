import numpy as np
import pickle

a = 1

for d in range(1, 6):
    for m in [5, 10, 20]:
        data = [np.zeros((1050,50)), np.zeros(1050)]
        with open("data_"+str(d)+"/data"+str(m)+".txt", "rb") as f:
            data_ = pickle.load(f)
        data[0][:1000,:] = data_[0][:1000]
        data[1][:1000] = data_[1][:1000]
        for i in range(1000, 1050):
            if np.random.rand() < 0.5:
                data[0][i,:m] = (np.random.rand(m) * a) - 2*a
                data[1][i] = 1
            else:
                data[0][i,:m] = (np.random.rand(m) * a) + a
                data[1][i] = -1
        data[0][1000:,m:] = (np.random.rand(50, 50-m) * 4*a) - 2*a
        with open("data_"+str(d)+"/data"+str(m)+".txt", "wb") as f:
            pickle.dump(data, f)
