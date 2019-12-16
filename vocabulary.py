import glob
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np


def extract_vocabulary(folder):
    files = glob.glob(folder+"*.txt")
    vocabulary = {}
    for filename in files:
        with open(filename) as f:
            text = f.readlines()
            for line in text:
                words = word_tokenize(line)
                for word in words:
                    word = word.lower()
                    if word in vocabulary.keys():
                        vocabulary[word] += 1
                    else:
                        vocabulary[word] = 1
    vocabulary.pop("", None)
    return vocabulary


def merge_vocabulary(old_vocab, new_vocab):
    for word in new_vocab.keys():
        if word in list(old_vocab.keys()):
            old_vocab[word] += new_vocab[word]
        else:
            old_vocab[word] = new_vocab[word]
    return old_vocab


def stop_word_removal(vocabulary):
    stop_words = set(stopwords.words('english'))
    for word in stop_words:
        vocabulary.pop(word, None)
    return vocabulary


def lemmatize(vocabulary):
    lemmatizer = WordNetLemmatizer()
    for word in list(vocabulary.keys()):
        lemmatized_word = lemmatizer.lemmatize(word)
        if lemmatized_word != word:
            try:
                vocabulary[lemmatized_word] += vocabulary[word]
            except KeyError:
                pass
            vocabulary.pop(word)
    return vocabulary


def threshold(vocabulary, lower_percentile, upper_percentile):
    lower_bound = np.percentile(np.fromiter(vocabulary.values(), dtype=float), lower_percentile)
    upper_bound = np.percentile(np.fromiter(vocabulary.values(), dtype=float), 100-upper_percentile)
    print(lower_bound, upper_bound)
    vocabulary = {key:value for key, value in vocabulary.items() if value > lower_bound and value < upper_bound}
    return vocabulary


if __name__ == '__main__':
    vocabulary = {}
    for i in range(1, 11):
        vocabulary = merge_vocabulary(vocabulary, extract_vocabulary("lingspam/part"+str(i)+"/"))
    vocabulary = stop_word_removal(vocabulary)
    vocabulary = lemmatize(vocabulary)
    vocabulary = threshold(vocabulary, 92, 0.15)
    with open("vocabulary.pickle", "wb") as f:
        pickle.dump(vocabulary, f)
