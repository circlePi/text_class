import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
import numpy as np

import config.config as config
from util.embedding_util import get_embedding

from net.attention import attention

torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
torch.cuda.manual_seed_all(2018)
np.random.seed(2018)

os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%config.device


class TextCNN(nn.Module):
    def __init__(self, vocab_size,
                 word_embedding_dimension,
                 word2id, filters,
                 kernel_size,
                 checkpoint_dir):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, word_embedding_dimension)
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.embedding.weight.data.copy_(torch.from_numpy(get_embedding(vocab_size, word_embedding_dimension, word2id)))

        self.conv0 = nn.Conv1d(in_channels=word_embedding_dimension,
                               out_channels=filters,
                               kernel_size=kernel_size[0])

        self.conv1 = nn.Conv1d(in_channels=word_embedding_dimension,
                               out_channels=filters,
                               kernel_size=kernel_size[1])

        self.conv2 = nn.Conv1d(in_channels=word_embedding_dimension,
                               out_channels=filters,
                               kernel_size=kernel_size[2])

        self.conv3 = nn.Conv1d(in_channels=word_embedding_dimension,
                               out_channels=filters,
                               kernel_size=kernel_size[3])

        self.linear = nn.Linear(in_features=32, out_features=9)

        self.batch_0 = nn.BatchNorm1d(num_features=12)
        self.batch_1 = nn.BatchNorm1d(num_features=32)

        self.checkpoint_dir = checkpoint_dir

    def k_max_pooling(self, x, dim=2, k=3):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def forward(self, x):
        batch_size = x.size(0)
        embeddings = self.embedding(x).permute(0, 2, 1)
        x0 = self.conv0(embeddings)
        x0 = F.relu(x0)
        x0 = self.k_max_pooling(x0)

        x1 = self.conv1(embeddings)
        x1 = F.relu(x1)
        x1 = self.k_max_pooling(x1)

        x2 = self.conv2(embeddings)
        x2 = F.relu(x2)
        x2 = self.k_max_pooling(x2)

        x3 = self.conv3(embeddings)
        x3 = F.relu(x3)
        x3 = self.k_max_pooling(x3)

        x = torch.cat((x0, x1, x2, x3), dim=2).permute(0, 2, 1)
        x = self.batch_0(x)
        x = attention(x, 1500)
        x = x.view(batch_size, -1)
        x = self.batch_1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        output = self.linear(x)
        return output

    def load(self, path):
        self.load_state_dict(torch.load(path))

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
        P = correct/y_pred.shape[0]
        return (P, f1)

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