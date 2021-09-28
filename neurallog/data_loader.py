import pandas as pd
import numpy as np
from collections import OrderedDict
import re

from sklearn.utils import shuffle
import pickle
import os
import gensim
import string
from transformers import GPT2Tokenizer, TFGPT2Model
from transformers import BertTokenizer, TFBertModel
from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf

#
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = TFGPT2Model.from_pretrained('gpt2')
#
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

#
xlm_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
xlm_model = TFRobertaModel.from_pretrained('roberta-base')

def gpt2_encoder(s, no_wordpiece=0):
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in gpt2_tokenizer.get_vocab().keys()]
        s = " ".join(words)
    try:
        inputs = gpt2_tokenizer(s, return_tensors='tf', max_length=512)
        outputs = gpt2_model(**inputs)
        v = tf.reduce_mean(outputs.last_hidden_state, 1)
        return v[0]
    except:
        return np.zeros((768,))


def bert_encoder(s, no_wordpiece=0):
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in bert_tokenizer.vocab.keys()]
        s = " ".join(words)
    inputs = bert_tokenizer(s, return_tensors='tf', max_length=512)
    outputs = bert_model(**inputs)
    v = tf.reduce_mean(outputs.last_hidden_state, 1)
    return v[0]


def xlm_encoder(s, no_wordpiece=0):
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in xlm_tokenizer.get_vocab().keys()]
        s = " ".join(words)
    inputs = xlm_tokenizer(s, return_tensors='tf', max_length=512)
    outputs = xlm_model(**inputs)
    v = tf.reduce_mean(outputs.last_hidden_state, 1)
    return v[0]


def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform'):
    (x_data, y_data) = shuffle(x_data, y_data)
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = train_pos
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    return (x_train, y_train), (x_test, y_test)


def slice_hdfs(x, y, window_size):
    results_data = []
    print("Slicing {} sessions, with window {}".format(x.shape[0], window_size))
    for idx, sequence in enumerate(x):
        seqlen = len(sequence)
        i = 0
        while (i + window_size) < seqlen:
            slice = sequence[i: i + window_size]
            results_data.append([idx, slice, sequence[i + window_size], y[idx]])
            i += 1
        else:
            slice = sequence[i: i + window_size]
            slice += ["#Pad"] * (window_size - len(slice))
            results_data.append([idx, slice, "#Pad", y[idx]])
    results_df = pd.DataFrame(results_data, columns=["SessionId", "EventSequence", "Label", "SessionLabel"])
    print("Slicing done, {} windows generated".format(results_df.shape[0]))
    return results_df[["SessionId", "EventSequence"]], results_df["Label"], results_df["SessionLabel"]


def clean(s):
    # s = re.sub(r'(\d+\.){3}\d+(:\d+)?', " ", s)
    # s = re.sub(r'(\/.*?\.[\S:]+)', ' ', s)
    s = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    content = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in content.strip().split()])
    return s


def load_HDFS(log_file, label_file=None, train_ratio=0.5, window='session',
              split_type='uniform',
              save_csv=False,
              window_size=0,
              e_type="bert"):
    """ Load HDFS structured log into train and test data
    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.
    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
    """

    print('====== Input data summary ======')

    E = {}
    if e_type == "bert":
        encoder = bert_encoder
    elif e_type == "xlm":
        encoder = xlm_encoder
    else:
        if e_type == "gpt2":
            encoder = gpt2_encoder
        else:
            encoder = word2vec_encoder

    E = {}
    t0 = time.time()
    if log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file, allow_pickle=True)
        x_data = data['data_x']
        y_data = data['data_y']
        (x_train, y_train), (x_test, y_test) = _split_data(x_data, y_data, train_ratio, split_type)

    elif log_file.endswith('.log'):
        assert window == 'session', "Only window=session is supported for HDFS dataset."
        print("Loading", log_file)
        with open(log_file, mode="r", encoding='utf8') as f:
            logs = f.readlines()
            logs = [x.strip() for x in logs]
        print("Loaded!")
        data_dict = OrderedDict()
        n_logs = len(logs)
        print(n_logs)
        for i, line in enumerate(logs):
            blkId_list = re.findall(r'(blk_-?\d+)', line)
            blkId_list = list(set(blkId_list))
            if len(blkId_list) >= 2:
                continue
            blkId_set = set(blkId_list)
            content = clean(line).lower()
            if content not in E.keys():
                E[content] = encoder(content, 0)
                print(content)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(E[content])
            i += 1
            if i % 1000 == 0 or i == n_logs:
                print("Loading {0:.2f} - number of unique message: {1}".format(i / n_logs * 100, len(E.keys())))
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
            print("Saving data...")
            np.savez_compressed("data-{0}.npz".format(e_type), data_x=data_df['EventSequence'].values,
                                data_y=data_df['Label'].values)
            # Split train and test data
            (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values,
                                                               data_df['Label'].values, train_ratio, split_type)

            print(y_train.sum(), y_test.sum())

        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)

        if window_size > 0:
            x_train, window_y_train, y_train = slice_hdfs(x_train, y_train, window_size)
            x_test, window_y_test, y_test = slice_hdfs(x_test, y_test, window_size)
            log = "{} {} windows ({}/{} anomaly), {}/{} normal"
            print(log.format("Train:", x_train.shape[0], y_train.sum(), y_train.shape[0], (1 - y_train).sum(),
                             y_train.shape[0]))
            print(log.format("Test:", x_test.shape[0], y_test.sum(), y_test.shape[0], (1 - y_test).sum(),
                             y_test.shape[0]))
            return (x_train, window_y_train, y_train), (x_test, window_y_test, y_test)

        if label_file is None:
            if split_type == 'uniform':
                split_type = 'sequential'
                print('Warning: Only split_type=sequential is supported \
                if label_file=None.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
            print('Total: {} instances, train: {} instances, test: {} instances'.format(
                x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_train, None), (x_test, None), data_df
    else:
        raise NotImplementedError('load_HDFS() only support csv and npz files!')
    print(time.time() - t0)

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test)

def balancing(x, y):
    if y.count(0) > y.count(1):
        pos_idx = [i for i, l in enumerate(y) if l == 1]
        neg_idx = [i for i, l in enumerate(y) if l == 0]
        pos_idx = shuffle(pos_idx)
        neg_idx = shuffle(neg_idx)
        neg_idx = neg_idx[:len(pos_idx) * 5]

        check_ids = [False] * len(x)
        for idx in pos_idx:
            check_ids[idx] = True

        for idx in neg_idx:
            check_ids[idx] = True

        x = [s for i, s in enumerate(x) if check_ids[i]]
        y = [s for i, s in enumerate(y) if check_ids[i]]
    else:
        pos_idx = [i for i, l in enumerate(y) if l == 1]
        neg_idx = [i for i, l in enumerate(y) if l == 0]
        pos_idx = shuffle(pos_idx)
        neg_idx = shuffle(neg_idx)
        pos_idx = pos_idx[:len(neg_idx)]

        check_ids = [False] * len(x)
        for idx in pos_idx:
            check_ids[idx] = True

        for idx in neg_idx:
            check_ids[idx] = True

        x = [s for i, s in enumerate(x) if check_ids[i]]
        y = [s for i, s in enumerate(y) if check_ids[i]]
    return x, y

def timestamp(log):
    log = log[log.find(" ") + 1:]
    t = log[:log.find(" ")]
    return float(t)

import time
def load_Supercomputers(log_file, train_ratio=0.5, windows_size=20, step_size=0, e_type='bert', e_name=None,
             mode="balance", NoWordPiece=0):
    print("Loading", log_file)

    with open(log_file, mode="r", encoding='utf8') as f:
        logs = f.readlines()
        logs = [x.strip() for x in logs]
    try:
        with open(e_name, mode='rb') as f:
            E = pickle.load(f)
    except:
        E = {}

    if e_type == "bert":
        encoder = bert_encoder
    elif e_type == "xlm":
        encoder = xlm_encoder
    else:
        if e_type == "gpt2":
            encoder = gpt2_encoder
        else:
            encoder = word2vec_encoder

    print("Loaded")
    x_tr, y_tr = [], []
    i = 0
    failure_count = 0
    n_train = int(len(logs) * train_ratio)
    c = 0
    t0 = time.time()
    while i < n_train - windows_size:
        c += 1
        if c % 1000 == 0:
            print("Loading {0:.2f} - {1} unique logs".format(i * 100 / n_train, len(E.keys())))
        if logs[i][0] != "-":
            failure_count += 1
        seq = []
        label = 0
        for j in range(i, i + windows_size):
            if logs[j][0] != "-":
                label = 1
            content = logs[j]
            # remove label from log messages
            content = content[content.find(' ') + 1:]
            content = clean(content.lower())
            if content not in E.keys():
                try:
                    E[content] = encoder(content, NoWordPiece)
                except:
                    print(content)
                # print(content)
            emb = E[content]
            seq.append(emb)
        x_tr.append(seq.copy())
        y_tr.append(label)
        # j = i + 1
        # while timestamp(logs[j]) - timestamp(logs[i]) < step_size:
        #     j += 1
        i = i + windows_size
    print("last train index:", i)
    x_te = []
    y_te = []
    #
    for i in range(n_train, len(logs) - windows_size, windows_size):
        if i % 1000 == 0:
            print("Loading {:.2f}".format(i * 100 / n_train))
        if logs[i][0] != "-":
            failure_count += 1
        seq = []
        label = 0
        for j in range(i, i + windows_size):
            if logs[j][0] != "-":
                label = 1
            content = logs[j]
            # remove label from log messages
            content = content[content.find(' ') + 1:]
            content = clean(content.lower())
            if content not in E.keys():
                E[content] = encoder(content)
                print(len(E.keys()))
            emb = E[content]
            seq.append(emb)
        x_te.append(seq.copy())
        # x_te.append(seq.copy())
        y_te.append(label)

    print(time.time() - t0)

    (x_tr, y_tr) = shuffle(x_tr, y_tr)
    print("Total failure logs: {0}".format(failure_count))

    if mode == 'balance':
        x_tr, y_tr = balancing(x_tr, y_tr)

    num_train = len(x_tr)
    num_test = len(x_te)
    num_total = num_train + num_test
    num_train_pos = sum(y_tr)
    num_test_pos = sum(y_te)
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_tr, y_tr), (x_te, y_te)


if __name__ == '__main__':
    # (x_tr, y_tr), (x_te, y_te) = load_Supercomputers(
    #     "../data/raw/BGL.log", train_ratio=0.8, windows_size=20,
    #     step_size=0, e_type='bert', e_name=None, mode='imbalance')
    #
    # with open("../data/embeddings/BGL/iforest-train.pkl", mode="wb") as f:
    #     pickle.dump((x_tr, y_tr), f, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open("../data/embeddings/BGL/iforest-test.pkl", mode="wb") as f:
    #     pickle.dump((x_te, y_te), f, protocol=pickle.HIGHEST_PROTOCOL)

    (x_tr, y_tr), (x_te, y_te) = load_HDFS(
        "../data/raw/HDFS/HDFS.log", "../data/raw/HDFS/anomaly_label.csv", train_ratio=0.8, split_type='sequential')
    #
    # with open("./data/embeddings/BGL/neural-train.pkl", mode="wb") as f:
    #     pickle.dump((x_tr, y_tr), f, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open("./data/embeddings/BGL/neural-test.pkl", mode="wb") as f:
    #     pickle.dump((x_te, y_te), f, protocol=pickle.HIGHEST_PROTOCOL)
