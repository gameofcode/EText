import re
import random
import numpy as np


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(anger_file, disgust_file, fear_file, happy_file, sad_file, surprise_file, neutral_file):
    '''
    :param anger_file:
    :param disgust_file:
    :param fear_file:
    :param happy_file:
    :param sad_file:
    :param surprise_file:
    :param neutral_file:
    :return:
    '''
    anger_data = list(open(anger_file, "r", encoding='utf-8').readlines())
    disgust_data = list(open(disgust_file, "r", encoding='utf-8').readlines())
    fear_data = list(open(fear_file, "r", encoding='utf-8').readlines())
    happy_data = list(open(happy_file, "r", encoding='utf-8').readlines())
    sad_data = list(open(sad_file, "r", encoding='utf-8').readlines())
    surprise_data = list(open(surprise_file, "r", encoding='utf-8').readlines())
    neutral_data = list(open(neutral_file, "r", encoding='utf-8').readlines())

    anger_data = random.sample([s.strip() for s in anger_data], 1000)
    disgust_data = random.sample([s.strip() for s in disgust_data], 1000)
    fear_data = random.sample([s.strip() for s in fear_data], 1000)
    happy_data = random.sample([s.strip() for s in happy_data], 1000)
    sad_data = random.sample([s.strip() for s in sad_data], 1000)
    surprise_data = random.sample([s.strip() for s in surprise_data], 1000)
    neutral_data = random.sample([s.strip() for s in neutral_data], 3000)

    # Split by words
    x_text = anger_data + disgust_data + fear_data + happy_data + sad_data + surprise_data + neutral_data
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    anger_lab = [[1, 0, 0, 0, 0, 0, 0] for _ in anger_data]
    disgust_lab = [[0, 1, 0, 0, 0, 0, 0] for _ in disgust_data]
    fear_lab = [[0, 0, 1, 0, 0, 0, 0] for _ in fear_data]
    happy_lab = [[0, 0, 0, 1, 0, 0, 0] for _ in happy_data]
    sad_lab = [[0, 0, 0, 0, 1, 0, 0] for _ in sad_data]
    surprise_lab = [[0, 0, 0, 0, 0, 1, 0] for _ in surprise_data]
    neutral_lab = [[0, 0, 0, 0, 0, 0, 1] for _ in neutral_data]
    y = np.concatenate([anger_lab, disgust_lab, fear_lab, happy_lab, sad_lab, surprise_lab, neutral_lab], 0)

    print("********* Dataset samples per class *********")
    print("anger \t\t: ", len(anger_data))
    print("disgust \t: ", len(disgust_data))
    print("fear \t\t: ", len(fear_data))
    print("happy \t\t: ", len(happy_data))
    print("sad \t\t: ", len(sad_data))
    print("surprise \t: ", len(surprise_data))
    print("neutral \t: ", len(neutral_data))
    print("*********************************************")

    print(len(x_text))
    print(len(y))
    print(x_text[0])
    print(y[0])
    print(x_text[-1])
    print(y[-1])
    print(x_text[:2])
    print(y[:2])

    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    print(data[0])
    print(data_size)
    print(num_batches_per_epoch)

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
