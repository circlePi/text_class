from preprocessing.data_processor import data_helper
from Io.data_loader import BatchIterator

from net.textcnn import TextCNN
from train.train import fit

import config.config as config
from util.porgress_util import ProgressBar

def text_cnn():
    # 数据预处理
    word2id, epoch_size = data_helper(vocab_size=config.vocab_size, max_len=config.max_len, min_freq=3, stop_list=None,
                valid_size=0.3, random_state=2018, shuffle=True, is_debug=config.is_debug)

    vocab_size = len(word2id)

    # 初始化进度条
    pbar = ProgressBar(epoch_size=epoch_size, batch_size=config.batch_size)

    # 加载batch
    bi = BatchIterator(config.TRAIN_FILE,
                       config.VALID_FILE,
                       config.batch_size, fix_length=config.max_len,
                       x_var="text", y_var=["label"],
                       format='tsv')
    train, valid = bi.create_dataset()
    train_iter, val_iter = bi.get_iterator(train, valid)

    # 初始化模型
    model = TextCNN(vocab_size=vocab_size,
                   word_embedding_dimension=config.word_embedding_dimension,
                   word2id=word2id,
                   filters=config.filters,
                   kernel_size=config.kernel_size,
                   checkpoint_dir=config.CHECKPOINT_DIR)

    # 训练
    fit(model, train_iter, val_iter,
        config.num_epoch, pbar,
        config.lr_decay_mode,
        config.initial_lr, verbose=1)
