import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np

import config.config as config
from util.embedding_util import get_embedding
from net.capsule import CapsNet

torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
torch.cuda.manual_seed_all(2018)
np.random.seed(2018)

os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%config.device


class TextCapsule(nn.Module):
    def __init__(self, vocab_size, word_embedding_dimension, word2id, checkpoint_dir):
        super(TextCapsule, self).__init__()

        self.embedding = nn.Embedding(vocab_size, word_embedding_dimension)
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.embedding.weight.data.copy_(torch.from_numpy(get_embedding(vocab_size, word_embedding_dimension, word2id)))

        self.capsule = CapsNet(300, 23*16)

        self.linear = nn.Linear(in_features=140, out_features=9)
        self.batch = nn.BatchNorm1d(num_features=140)

        self.checkpoint_dir = checkpoint_dir


    def forward(self, x):
        embeddings = self.embedding(x)  # (64, T, 300)
        caps_output = self.capsule(embeddings)
        output = self.batch(caps_output)
        output = F.dropout(output, p=0.5, training=self.training)
        output = self.linear(output)
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
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
        print('\nclassify_report:\n', classify_report)
        print('\nconfusion_matrix:\n', conf_matrix)