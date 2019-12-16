import numpy as np
import pickle
import matplotlib.pyplot as plt


def rho(i):
    return 0.0004 / (100 + i)


if __name__ == '__main__':
    for d in range(1, 4):
        print(d)
        for m in [5, 10, 20]:
            with open("data_"+str(d)+"/data"+str(m)+".txt", "rb") as f:
                data = pickle.load(f)
            w = (np.random.rand(50) * 0.02) - 0.01
            converged = False
            epochs = 0
            while not(converged) and epochs < 300:
                epochs += 1
                del_w = np.zeros(w.shape)
                mispredictions = 0
                l_rate = rho(epochs)
                for i in range(1000):
                    if data[1][i] * np.dot(data[0][i,:], w) < 0:
                        mispredictions += 1
                        del_w += data[1][i] * data[0][i,:]
                w += l_rate * del_w
                if mispredictions == 0:
                    converged = True

            print(m, epochs)

            np.savetxt("data_"+str(d)+"/batch_and_increment_variable_"+str(m)+".csv", w, delimiter=",")
