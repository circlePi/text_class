import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn.metrics import f1_score
import numpy as np

import config.config as config
from util.embedding_util import get_embedding

torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
torch.cuda.manual_seed_all(2018)
np.random.seed(2018)

os.environ["CUDA_VISIBLE_DEVICE"] = "%d"%config.device

class RNN(nn.Module):
    def __init__(self, vocab_size,
                 word_embedding_dimension,
                 word2id,
                 hidden_size, bi_flag,
                 num_layer,
                 labels,
                 cell_type,
                 dropout,
                 checkpoint_dir):
        super(RNN, self).__init__()
        self.labels = labels
        self.num_label = len(labels)
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.checkpoint_dir = checkpoint_dir

        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.embedding = nn.Embedding(vocab_size, word_embedding_dimension)
        for p in self.embedding.parameters():
            p.requires_grad = False
        self.embedding.weight.data.copy_(torch.from_numpy(get_embedding(vocab_size,
                                                                        word_embedding_dimension,
                                                                        word2id)))

        if cell_type == "LSTM":
            self.rnn_cell = nn.LSTM(input_size=word_embedding_dimension,
                                    hidden_size=hidden_size,
                                    num_layers=num_layer,
                                    batch_first=True,
                                    dropout=dropout,
                                    bidirectional=bi_flag)
        elif cell_type == "GRU":
            self.rnn_cell = nn.GRU(input_size=word_embedding_dimension,
                                   hidden_size=hidden_size,
                                   num_layers=num_layer,
                                   batch_first=True,
                                   dropout=dropout,
                                   bidirectional=bi_flag)
        else:
            raise TypeError("RNN: Unknown rnn cell type")

        # 是否双向
        self.bi_num = 2 if bi_flag else 1

        self.linear = nn.Linear(hidden_size*self.bi_num, self.num_label)

    def forward(self, inputs, length):
        batch_size = inputs.shape[0]

        embeddings = self.embedding(inputs)  # (batch_size, time_steps, embedding_dim)
        # 去除padding元素
        # embeddings_packed: (batch_size*time_steps, embedding_dim)
        embeddings_packed = pack_padded_sequence(embeddings, length, batch_first=True)
        output, (h_n, c_n) = self.rnn_cell(embeddings_packed)
        # padded_output: (batch_size, time_steps, hidden_size * bi_num)
        # h_n|c_n: (num_layer*bi_num, batch_size, hidden_size)
        padded_output, len = pad_packed_sequence(output, batch_first=True)
        # 取最后一个有效输出作为最终输出（0为无效输出）
        last_output = padded_output[torch.LongTensor(range(batch_size)), length-1]

        last_output = F.dropout(last_output, p=self.dropout, training=self.training)
        output = self.linear(last_output)
        return output

    def load(self):
        self.load_state_dict(torch.load(self.checkpoint_dir))

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_dir)

    def evaluate(self, y_pred, y_true):
        _, y_pred = torch.max(y_pred.data, 1)
        if config.use_cuda:
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().numpy()
        else:
            y_true = y_true.numpy()
            y_pred = y_pred.numpy()
        f1 = f1_score(y_true, y_pred, labels=self.labels, average="macro")
        correct = np.sum((y_true==y_pred).astype(int))
        acc = correct/y_pred.shape[0]
        return (acc, f1)

