import os
import random
import pickle
import operator
from glob import glob
from tqdm import tqdm
from collections import Counter

import config.config as config
from util.Logginger import init_logger

logger = init_logger("torch", logging_path=config.LOG_PATH)


def sent_label_split(line):
    """
    句子处理成单词
    :param line: 原始行
    :return: 单词， 标签
    """
    line = line.strip('\n').split('@')
    label = line[0]
    sent = line[1].split(' ')
    return sent, label

def word_to_id(word, word2id):
    """
    单词-->ID
    :param word: 单词
    :param word2id: word2id @type: dict
    :return:
    """
    return word2id[word] if word in word2id else word2id[config.flag_words[1]]


def bulid_vocab(vocab_size, min_freq=3, stop_word_list=None,
                is_debug=False):
    """
    建立词典
    :param vocab_size: 词典大小
    :param min_freq: 最小词频限制
    :param stop_list: 停用词 @type：file_path
    :param is_debug: 是否测试模式 @type: bool True:使用很小的数据集进行代码测试
    :return: word2id
    """
    size = 0
    count = Counter()

    with open(os.path.join(config.ROOT_DIR, config.RAW_DATA), 'r') as fr:
        logger.info('Building vocab')
        for line in tqdm(fr, desc='Build vocab'):
            words, label = sent_label_split(line)
            count.update(words)
            size += 1
            if is_debug:
                limit_train_size = 5000
                if size > limit_train_size:
                    break
    if stop_word_list:
        stop_list = {}
        with open(os.path.join(config.ROOT_DIR, config.STOP_WORD_LIST), 'r') as fr:
                for i, line in enumerate(fr):
                    word = line.strip('\n')
                    if stop_list.get(word) is None:
                        stop_list[word] = i
        count = {k: v for k, v in count.items() if k not in stop_list}
    count = sorted(count.items(), key=operator.itemgetter(1))
    # 词典
    vocab = [w[0] for w in count if w[1] >= min_freq]
    if vocab_size-2 < len(vocab):
        vocab = vocab[:vocab_size-2]
    vocab = config.flag_words + vocab
    logger.info('vocab_size is %d'%len(vocab))
    # 词典到编号的映射
    word2id = {k: v for k, v in zip(vocab, range(0, len(vocab)))}
    assert word2id['<pad>'] == 0, "ValueError: '<pad>' id is not 0"
    # print(word2id)
    with open(config.WORD2ID_FILE, 'wb') as fw:
        pickle.dump(word2id, fw)
    return word2id


def train_val_split(X, y, valid_size=0.3, random_state=2018, shuffle=True):
    """
    训练集验证集分割
    :param X: sentences
    :param y: labels
    :param random_state: 随机种子
    """
    logger.info('train val split')

    train, valid = [], []
    bucket = [[] for _ in config.labels]

    for data_x, data_y in tqdm(zip(X, y), desc='bucket'):
        bucket[int(data_y)].append((data_x, data_y))

    del X, y

    for bt in tqdm(bucket, desc='split'):
        N = len(bt)
        if N == 0:
            continue
        test_size = int(N * valid_size)

        if shuffle:
            random.seed(random_state)
            random.shuffle(bt)

        valid.extend(bt[:test_size])
        train.extend(bt[test_size:])

    if shuffle:
        random.seed(random_state)
        random.shuffle(valid)
        random.shuffle(train)

    return train, valid


def text2id(word2id, maxlen=None, valid_size=0.3, random_state=2018, shuffle=True, is_debug=False):
    """
    训练集文本转ID
    :param valid_size: 验证集大小
    """
    print(os.path.join(config.ROOT_DIR, config.TRAIN_FILE))
    file_name = os.path.join(config.ROOT_DIR, config.TRAIN_FILE)
    if len(glob(file_name)) > 0:
        logger.info('Text to id file existed')
        epoch_size = int(os.popen('cat %s | wc -l'%file_name).readlines()[0].strip('\n'))
        return epoch_size
    logger.info('Text to id')
    sentences, labels, lengths = [], [], []
    size = 0
    with open(os.path.join(config.ROOT_DIR, config.RAW_DATA), 'r') as fr:
        for line in tqdm(fr, desc='text_to_id'):
            words, label = sent_label_split(line)
            sent = [word_to_id(word=word, word2id=word2id) for word in words]
            if maxlen:
                sent = sent[:maxlen]
            length = len(sent)
            sentences.append(sent)
            labels.append(label)
            lengths.append(length)
            size += 1
            if is_debug:
                limit_train_size = 5000
                if size > limit_train_size:
                    break

    train, valid = train_val_split(sentences, labels,
                                   valid_size=valid_size,
                                   random_state=random_state,
                                   shuffle=shuffle)
    epoch_size = len(train)

    del sentences, labels, lengths


    with open(config.TRAIN_FILE, 'w') as fw:
        for sent, label in train:
            sent = [str(s) for s in sent]
            line = "\t".join([str(label), " ".join(sent)])
            fw.write(line + '\n')
        logger.info('Writing train to file done')

    with open(config.VALID_FILE, 'w') as fw:
        for sent, label in valid:
            sent = [str(s) for s in sent]
            line = str(label) + '\t' + " ".join(sent)
            fw.write(line + '\n')
        logger.info('Writing valid to file done')
    return epoch_size


def data_helper(vocab_size, max_len, min_freq=3, stop_list=None,
                valid_size=0.3, random_state=2018, shuffle=True, is_debug=False):
    # 判断文件是否已存在
    if len(glob(os.path.join(config.ROOT_DIR, config.WORD2ID_FILE))) > 0:
        logger.info('Word to id file existed')
        with open(os.path.join(config.ROOT_DIR, config.WORD2ID_FILE), 'rb') as fr:
            word2id = pickle.load(fr)
    else:
        word2id = bulid_vocab(vocab_size=vocab_size, min_freq=min_freq, stop_word_list=stop_list,
                is_debug=is_debug)
    epoch_size = text2id(word2id, valid_size=valid_size, maxlen=max_len, random_state=random_state, shuffle=shuffle, is_debug=is_debug)
    return word2id, epoch_size







