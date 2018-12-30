from tqdm import tqdm
import numpy as np
import config.config as config
from util.Logginger import init_logger

logger = init_logger("torch", logging_path=config.LOG_PATH)


def parse_word_vector(word_index):
    pre_trained_wordvector = {}
    with open(config.EMBEDDING_FILE, 'r') as fr:
        for line in fr:
            lines = line.strip('\n').split(' ')
            word = lines[0]
            if word_index.get(word) is not None:
                vector = lines[1:]
                pre_trained_wordvector[word] = vector
            else:
                continue
        return pre_trained_wordvector


def get_embedding(vocab_size, embedding_dim, word2id):
    logger.info('Get embedding')
    pre_trained_wordector = parse_word_vector(word2id)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    for word, id in tqdm(word2id.items()):
        try:
            word_vector = pre_trained_wordector[word]
            embedding_matrix[id] = word_vector
        except:
            continue
    logger.info('Get embedding done')
    return embedding_matrix