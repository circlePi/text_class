import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
import numpy as np

from util.embedding_util import get_embedding
import config.config as config

from net.reinforced_cnn import Reforced_CNN
from net.reinforced_gru import Reinforced_GRU

torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
torch.cuda.manual_seed_all(2018)
np.random.seed(2018)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Hybrid(nn.Module):
    def __init__(self,
                 vocab_size,
                 word_embedding_dimension,
                 word2id,
                 dropout,
                 attention_size,
                 filters, kernel_size,
                 hidden_size, bi_flag, num_layer,
                 checkpoint_dir):
        super(Hybrid, self).__init__()

        self.checkpoint_dir = checkpoint_dir

        self.embedding = nn.Embedding(vocab_size, word_embedding_dimension)
        for p in self.embedding.parameters():
            p.requires_grad = False
        self.embedding.weight.data.copy_(torch.from_numpy(get_embedding(vocab_size, word_embedding_dimension, word2id)))

        self.projection = nn.Sequential(
            nn.Linear(in_features=568, out_features=256),  # 537
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=9),
            nn.BatchNorm1d(num_features=9),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.dropout = dropout
        self.ref_cnn = Reforced_CNN(word_embedding_dimension,
                                    filters,
                                    kernel_size,
                                    attention_size)
        self.ref_gru = Reinforced_GRU(word_embedding_dimension,
                                      hidden_size,
                                      bi_flag,
                                      num_layer,
                                      attention_size)

    def forward(self, inputs):
        # (batch_size, time_steps, embedding_dim)
        embeddings = self.embedding(inputs)
        # (batch_size, num_capsule*out_channel+1)
        ref_cnn_output = self.ref_cnn(embeddings)
        # (batch_size, num_capsule*out_channel+256)
        ref_gru_output = self.ref_gru(embeddings)
        out = torch.cat((ref_cnn_output, ref_gru_output), dim=1)
        # out = ref_cnn_output
        output = self.projection(out)
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
        f1 = f1_score(y_true, y_pred, labels=config.labels, average="macro")
        correct = np.sum((y_true==y_pred).astype(int))
        acc = correct/y_pred.shape[0]
        return (acc, f1)

    def class_report(self, y_pred, y_true):
        _, y_pred = torch.max(y_pred.data, 1)
        if config.use_cuda:
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().numpy()
        else:
            y_true = y_true.numpy()
            y_pred = y_pred.numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)





