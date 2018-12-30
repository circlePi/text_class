"""将id格式的输入转换成dataset，并做动态padding"""

import torch
from torchtext.data import Field, TabularDataset
from torchtext.data import BucketIterator

import config.config as config

def x_tokenize(x):
    # 如果加载进来的是已经转成id的文本
    # 此处必须将字符串转换成整型
    return [int(c) for c in x.split()]

def y_tokenize(y):
    return int(y)

class BatchIterator(object):
    def __init__(self, train_path, valid_path,
                 batch_size, fix_length=None,
                 x_var="text", y_var=["label"],
                 format='tsv'):
        self.train_path = train_path
        self.valid_path = valid_path
        self.batch_size = batch_size
        self.fix_length = fix_length
        self.format = format
        self.x_var = x_var
        self.y_vars = y_var

    def create_dataset(self):
        TEXT = Field(sequential=True, tokenize=x_tokenize,
                     use_vocab=False, batch_first=True,
                     fix_length=self.fix_length,   #  如需静态padding,则设置fix_length, 但要注意要大于文本最大长度
                     eos_token=None, init_token=None,
                     include_lengths=True, pad_token=0)
        LABEL = Field(sequential=False, tokenize=y_tokenize, use_vocab=False, batch_first=True)

        fields = [
            ("label", LABEL), ("text", TEXT)]

        train, valid = TabularDataset.splits(
            path=config.ROOT_DIR,
            train=self.train_path, validation=self.valid_path,
            format='tsv',
            skip_header=False,
            fields=fields)
        return train, valid


    def get_iterator(self, train, valid):
        train_iter, val_iter = BucketIterator.splits((train, valid),
                                                     batch_sizes=(self.batch_size, self.batch_size),
                                                     device = torch.device("cpu"),  # cpu by -1, gpu by 0
                                                     sort_key=lambda x: len(x.text), # field sorted by len
                                                     sort_within_batch=True,
                                                     repeat=False)

        train_iter = BatchWrapper(train_iter, x_var=self.x_var, y_vars=self.y_vars)
        val_iter = BatchWrapper(val_iter, x_var=self.x_var, y_vars=self.y_vars)
        ### batch = iter(train_iter)
        ### batch： ((text, length), y)
        return train_iter, val_iter



class BatchWrapper(object):
    """对batch做个包装，方便调用，可选择性使用"""
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)

            if self.y_vars is not None:
                temp = [getattr(batch, feat).unsqueeze(1) for feat in self.y_vars]
                y = torch.cat(temp, dim=1).long()
            else:
                raise ValueError('BatchWrapper: invalid label')
            text = x[0]
            length = x[1]
            yield (text, y, length)
            # yield (x, y)

    def __len__(self):
        return len(self.dl)








