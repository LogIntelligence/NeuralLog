import pandas as pd
import numpy as np
from collections import OrderedDict
import re

from sklearn.utils import shuffle
import pickle
import string
from transformers import GPT2Tokenizer, TFGPT2Model
from transformers import BertTokenizer, TFBertModel
from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf
import time

# Pre-trained GPT2 model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = TFGPT2Model.from_pretrained('gpt2')

# Pre-trained BERT model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Pre-trained XLM model
xlm_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
xlm_model = TFRobertaModel.from_pretrained('roberta-base')


def gpt2_encoder(s, no_wordpiece=0):
    """ Compute semantic vectors with GPT2
    Parameters
    ----------
    s: string to encode
    no_wordpiece: 1 if you do not use sub-word tokenization, otherwise 0

    Returns
    -------
        np array in shape of (768,)
    """
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in gpt2_tokenizer.get_vocab().keys()]
        s = " ".join(words)
    try:
        inputs = gpt2_tokenizer(s, return_tensors='tf', max_length=512)
        outputs = gpt2_model(**inputs)
        v = tf.reduce_mean(outputs.last_hidden_state, 1)
        return v[0]
    except Exception as _:
        return np.zeros((768,))


def bert_encoder(s, no_wordpiece=0):
    """ Compute semantic vector with BERT
    Parameters
    ----------
    s: string to encode
    no_wordpiece: 1 if you do not use sub-word tokenization, otherwise 0

    Returns
    -------
        np array in shape of (768,)
    """
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in bert_tokenizer.vocab.keys()]
        s = " ".join(words)
    inputs = bert_tokenizer(s, return_tensors='tf', max_length=512)
    outputs = bert_model(**inputs)
    v = tf.reduce_mean(outputs.last_hidden_state, 1)
    return v[0]


def xlm_encoder(s, no_wordpiece=0):
    """ Compute semantic vector with XLM
    Parameters
    ----------
    s: string to encode
    no_wordpiece: 1 if you do not use sub-word tokenization, otherwise 0

    Returns
    -------
        np array in shape of (768,)
    """
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in xlm_tokenizer.get_vocab().keys()]
        s = " ".join(words)
    inputs = xlm_tokenizer(s, return_tensors='tf', max_length=512)
    outputs = xlm_model(**inputs)
    v = tf.reduce_mean(outputs.last_hidden_state, 1)
    return v[0]


def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform'):
    """ Split train/test data
    Parameters
    ----------
    x_data: list, set of log sequences (in the type of semantic vectors)
    y_data: list, labels for each log sequence
    train_ratio: float, training ratio (e.g., 0.8)
    split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.
    Returns
    -------

    """
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


def clean(s):
    """ Preprocess log message
    Parameters
    ----------
    s: str, raw log message

    Returns
    -------
    str, preprocessed log message without number tokens and special characters
    """
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
              split_type='uniform', e_type="bert", no_word_piece=0):
    """ Load HDFS unstructured log into train and test data
    Arguments
    ---------
        log_file: str, the file path of raw log (extension: .log).
        label_file: str, the file path of anomaly labels (extension: .csv).
        train_ratio: float, the ratio of training data for train/test split.
        window: str, the window options including `session` (default).
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.
        e_type: str, embedding type (choose from BERT, XLM, and GPT2).
        no_word_piece: bool, use split word into wordpiece or not.
    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
    """

    print('====== Input data summary ======')

    e_type = e_type.lower()
    if e_type == "bert":
        encoder = bert_encoder
    elif e_type == "xlm":
        encoder = xlm_encoder
    else:
        if e_type == "gpt2":
            encoder = gpt2_encoder
        else:
            raise ValueError('Embedding type {0} is not in BERT, XLM, and GPT2'.format(e_type.upper()))

    E = {}
    t0 = time.time()
    assert log_file.endswith('.log'), "Missing .log file"
    # elif log_file.endswith('.log'):
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    print("Loading", log_file)
    with open(log_file, mode="r", encoding='utf8') as f:
        logs = f.readlines()
        logs = [x.strip() for x in logs]
    data_dict = OrderedDict()
    n_logs = len(logs)
    print(n_logs)
    print("Loaded", n_logs, "lines!")
    for i, line in enumerate(logs):
        blkId_list = re.findall(r'(blk_-?\d+)', line)
        blkId_list = list(set(blkId_list))
        if len(blkId_list) >= 2:
            continue
        blkId_set = set(blkId_list)
        content = clean(line).lower()
        if content not in E.keys():
            E[content] = encoder(content, no_word_piece)
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = []
            data_dict[blk_Id].append(E[content])
        i += 1
        if i % 1000 == 0 or i == n_logs:
            print("\rLoading {0:.2f}% - number of unique message: {1}".format(i / n_logs * 100, len(E.keys())), end="")
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
    else:
        raise NotImplementedError("Missing label file for the HDFS dataset!")

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
    # else:
    #     raise NotImplementedError('load_HDFS() only support csv and npz files!')
    print("\nLoaded all HDFS dataset in: ", time.time() - t0)

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


def load_supercomputers(log_file, train_ratio=0.5, windows_size=20, step_size=0, e_type='bert', mode="balance",
                        no_word_piece=0):
    """ Load BGL, Thunderbird, and Spirit unstructured log into train and test data
    Parameters
    ----------
    log_file: str, the file path of raw log (extension: .log).
    train_ratio: float, the ratio of training data for train/test split.
    windows_size: int, the window size for sliding window
    step_size: int, the step size for sliding window. if step_size is equal to window_size then fixed window is applied.
    e_type: str, embedding type (choose from BERT, XLM, and GPT2).
    mode: str, split train/testing in balance or not
    no_word_piece: bool, use split word into wordpiece or not.

    Returns
    -------
    (x_tr, y_tr): the training data
    (x_te, y_te): the testing data
    """
    print("Loading", log_file)

    with open(log_file, mode="r", encoding='utf8') as f:
        logs = f.readlines()
        logs = [x.strip() for x in logs]
    E = {}
    e_type = e_type.lower()
    if e_type == "bert":
        encoder = bert_encoder
    elif e_type == "xlm":
        encoder = xlm_encoder
    else:
        if e_type == "gpt2":
            encoder = gpt2_encoder
        else:
            raise ValueError('Embedding type {0} is not in BERT, XLM, and GPT2'.format(e_type.upper()))

    print("Loaded", len(logs), "lines!")
    x_tr, y_tr = [], []
    i = 0
    failure_count = 0
    n_train = int(len(logs) * train_ratio)
    c = 0
    t0 = time.time()
    while i < n_train - windows_size:
        c += 1
        if c % 1000 == 0:
            print("\rLoading {0:.2f}% - {1} unique logs".format(i * 100 / n_train, len(E.keys())), end="")
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
                    E[content] = encoder(content, no_word_piece)
                except Exception as _:
                    print(content)
            emb = E[content]
            seq.append(emb)
        x_tr.append(seq.copy())
        y_tr.append(label)
        i = i + step_size
    print("\nlast train index:", i)
    x_te = []
    y_te = []
    #
    for i in range(n_train, len(logs) - windows_size, step_size):
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
                E[content] = encoder(content, no_word_piece)
            emb = E[content]
            seq.append(emb)
        x_te.append(seq.copy())
        y_te.append(label)

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
