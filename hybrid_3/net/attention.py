import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import config.config as config

os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%config.device

# def _parameter_init(param, mean=0, std=0.05):
#     param.data.normal_(mean, std)


def matrix_mul(inputs, weight, bias=None):
    feature_list = []
    for input in inputs:
        feature = torch.mm(input, weight)  # (T, C)*(C, A) = (T, A)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature)  # 归一化十分重要
        feature_list.append(feature)
    return torch.stack(feature_list, 0).squeeze()  # (B, T)


def wise_mul(inputs, alphas):
    feature_list = []
    for sequence, alpha in zip(inputs, alphas):
        alpha = alpha.unsqueeze(1)
        feature = sequence * alpha
        feature_list.append(feature)
    output = torch.stack(feature_list, 0)
    return torch.sum(output, 1)


def attention(inputs, attention_size):
    """
    :param inputs: (batch_size, time_steps, hidden_size)
    """
    hidden_size = inputs.shape[2]
    w = nn.Parameter(torch.randn(hidden_size, attention_size))
    b = nn.Parameter(torch.randn(1, attention_size))
    u = nn.Parameter(torch.randn(attention_size, 1))

    if config.use_cuda:
        w = w.cuda()
        b = b.cuda()
        u = u.cuda()

    v = matrix_mul(inputs, w, b)       # (B, T, A)
    u_v = matrix_mul(v,u)              # (B, T)
    alphas = F.softmax(u_v)            # (B, T)
    output = wise_mul(inputs, alphas)  # (B, H)
    return output







