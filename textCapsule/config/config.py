# ---------PATH------------
ROOT_DIR = '/home/daizelin/textCapsule/'
RAW_DATA = 'data/police_train.csv'

EMBEDDING_FILE = 'embedding/taizhou_min_count_1_window_5_300d.word2vec'
TRAIN_FILE = 'output/intermediate/train.tsv'
WORD2ID_FILE = 'output/intermediate/word2id.pkl'
VALID_FILE = 'output/intermediate/valid.tsv'
LOG_PATH = 'output/logs'
STOP_WORD_LIST = 'data/stop_list_chn.txt'
CHECKPOINT_DIR = 'output/checkpoints/text_capsule.ckpt'


# ---------DATA PARAM--------------
is_debug = False
flag_words = ['<pad>', '<unk>']
max_len = 25

# ------------NET PARAM------------
seed = 2018
device = 0
labels = range(9)
plot_path = 'output/img/loss_acc.jpg'
### ----------ATTENTION------------
attention_size = 1500
### ----------CAPSULE--------------
### ----------HYBRID---------------
#### -------REINFORCED_CNN---------
vocab_size = 1000000
word_embedding_dimension = 300
filters = 32
kernel_size = [1, 2, 3, 4]
#### -------REINFORCED_GRU---------
hidden_size = 128
bi_flag = True
num_layer = 1


### -------TRAIN-------------
num_epoch = 4
batch_size = 128
initial_lr = 0.001
lr_decay_mode = "custom_decay"
use_cuda = True
use_mem_track = False
