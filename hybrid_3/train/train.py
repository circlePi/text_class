import os
import time
import inspect
import torch
import torch.nn as nn
from torch.autograd import Variable

import config.config as config
from util.gpu_mem_track import MemTracker
from util.plot_util import loss_acc_plot
from util.lr_util import lr_update
from util.Logginger import init_logger

logger = init_logger("torch", logging_path=config.LOG_PATH)

torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
torch.cuda.manual_seed_all(2018)


import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%config.device


frame = inspect.currentframe()
gpu_tracker = MemTracker(frame)
use_cuda = config.use_cuda if torch.cuda.is_available() else False


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data)


def fit(model, training_iter, eval_iter, num_epoch, pbar, lr_decay_mode, initial_lr, verbose=1):
    model.apply(weights_init)

    if use_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    eval_losses = []
    train_accuracy = []
    eval_accuracy = []

    history = {
        "train_loss": train_losses,
        "train_acc": train_accuracy,
        "eval_loss": eval_losses,
        "eval_acc": eval_accuracy
    }

    best_f1 = 0

    start = time.time()
    for e in range(num_epoch):
        if e > 0:
            lr_update(optimizer=optimizer, epoch=e, lr_decay_mode=lr_decay_mode)

        model.train()
        for index, (inputs, label) in enumerate(training_iter):
            if config.use_mem_track:
                gpu_tracker.track()
            if use_cuda:
                inputs = Variable(inputs.cuda())
                label = Variable(label.squeeze(1).cuda())

            y_preds = model(inputs)
            train_loss = loss_fn(y_preds, label)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_acc, _ = model.evaluate(y_preds, label)
            pbar.show_process(train_acc, train_loss.data, time.time()-start, index)

            if config.use_mem_track:
                gpu_tracker.track()

        if use_cuda:
            torch.cuda.empty_cache()

        model.eval()
        count = 0
        y_predicts, y_labels = [], []
        eval_loss, eval_acc, eval_f1 = 0, 0, 0
        with torch.no_grad():
            for eval_inputs, eval_label in eval_iter:
                if use_cuda:
                    eval_inputs, eval_label = eval_inputs.cuda(), eval_label.squeeze(1).cuda()
                eval_y_preds = model(eval_inputs)
                eval_loss += loss_fn(eval_y_preds, eval_label).data
                y_predicts.append(eval_y_preds)
                y_labels.append(eval_label)
                count += 1
            eval_predicted = torch.cat(y_predicts, dim=0)
            eval_labeled = torch.cat(y_labels, dim=0)

            eval_acc, eval_f1 = model.evaluate(eval_predicted, eval_labeled)
            model.class_report(eval_predicted, eval_labeled)


            logger.info(
                '\n\nEpoch %d - train_loss: %4f - eval_loss: %4f - train_acc:%4f - eval_acc:%4f - eval_f1:%4f\n'
                % (e + 1,
                   train_loss.data,
                   eval_loss/count,
                   train_acc,
                   eval_acc,
                   eval_f1))

            # 保存最好的模型
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                model.save()

            if e % verbose == 0:
                train_losses.append(train_loss.data)
                train_accuracy.append(train_acc)
                eval_losses.append(eval_loss/count)
                eval_accuracy.append(eval_acc)
    loss_acc_plot(history)