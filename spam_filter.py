import glob
import pickle
import numpy as np
from pocket import train_perceptron
from least_mean_square import lms_update, lms_condition
import matplotlib.pyplot as plt
from multiprocessing import Pool


def rho(i):
    return 0.01 / (0.1 + i)


def read_file(filename):
    with open(filename, "rb") as f:
        return [pickle.load(f), -2*int("spm" in filename) + 1]


def read_data(folder):
    with open("vocabulary.pickle", "rb") as f:
        data = [np.ones((len(folder), len(pickle.load(f))+1)), np.zeros(len(folder))]
    pool = Pool()
    temp = [None for i in range(len(folder))]
    for i, filename in enumerate(folder):
        temp[i] = pool.apply_async(read_file, (filename, ))
    pool.close()
    pool.join()
    for i, val in enumerate(temp):
        data[0][i,:-1] = val.get()[0]
        data[1][i] = val.get()[1]
    return data


def test(folder, weight_filename):
    data = read_data(folder)
    w = np.loadtxt(weight_filename, delimiter=",")
    predictions = np.greater(np.dot(data[0], w), np.zeros(len(data[0])))
    TP, FP, TN, FN = 0, 0, 0, 0
    for prediction, label in zip(predictions, data[1]):
        if prediction:
            if label == 1:
                TN += 1
            else:
                FN += 1
        else:
            if label == 1:
                FP += 1
            else:
                TP += 1
    print(TP, ",", FP, ",", TN, ",", FN)
    return (TP+TN)/(TP+TN+FP+FN)


if __name__ == '__main__':
    data = read_data(glob.glob("lingspam/lingspam_full/train_tfidf/*.pickle"))
    pool = Pool()
    epochs = 100
    n = 6
    y = [None for i in range(n)]
    for i in range(n):
        weight_filename = "spam_filter_weights_" + str(i) + ".csv"
        y[i] = pool.apply_async(train_perceptron, (data, epochs, weight_filename, int(np.random.rand()*1000), rho, lms_update, lms_condition))
    pool.close()
    pool.join()
    for i in range(n):
        y[i] = y[i].get()
        plt.plot(y[i], label=str(i))
    plt.legend()
    plt.show()
    train_accuracy= 0
    test_accuracy = 0
    for i in range(n):
        weight_filename = "spam_filter_weights_" + str(i) + ".csv"
        test_accuracy += test(glob.glob("lingspam/lingspam_full/test_tfidf/*.pickle"), weight_filename)
        train_accuracy += y[i-1][-1]
    train_accuracy = 1 - train_accuracy/(n*len(data[1]))
    test_accuracy /= n
    print(train_accuracy, test_accuracy)
