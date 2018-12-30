import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import config.config as config


class CapsNet(nn.Module):
    def __init__(self, in_channel, num_route):
        super(CapsNet, self).__init__()
        self.conv = ConvLayer(in_channels=in_channel)
        self.primary_caps = PrimaryCaps()
        self.digit_caps = DigitCaps(num_route)

    def forward(self, input):
        batch_size = input.size(0)
        # (B, T, 1) -> (B, T-2, 19)
        x = F.relu(self.conv(input))
        # (B, T-2, 19) -> (B, T-4, 16)
        caps_1 = self.primary_caps(x)
        # (B, T-4, 16) -> (B, 10, 14, 1)
        caps_2 = self.digit_caps(caps_1)
        output = caps_2.view(batch_size, -1)
        return output


class ConvLayer(nn.Module):
    """
    input: for ref_cnn (B, T, 1), for ref_gru (B, T, 256)
    output: (B, T-1, 19)
    """
    def __init__(self,
                 in_channels,
                 out_channels=19,
                 kernel_size=2,
                 stride=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride)
    def forward(self, x):
        return F.relu(self.conv(x.permute(0, 2, 1)).permute(0, 2, 1))


class PrimaryCaps(nn.Module):
    """
    input: (B, T-1, 19)
    output: (B, T-2, 16)
    """
    def __init__(self,
                 in_channels=19,
                 out_channels=16,
                 kernel_size=2,
                 num_capsule = 5,
                 stride=1):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList(
            [nn.Conv1d(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       stride=stride)
             for _ in range(num_capsule)])
        self.num_capsule = num_capsule

    def forward(self, x):
         # [(B, T-2, 16)] * num_capsule
         u = [capsule(x.permute(0, 2, 1)).permute(0, 2, 1) for capsule in self.capsules]
         u = torch.stack(u, dim=1)
         u = u.view(x.size(0), -1, self.num_capsule)  # (B, (T-2)*16, 5)==(batch_size, features, num_capsule)
         return squash(u)


class DigitCaps(nn.Module):
    """
    将(T-4)*16个1*5的向量映射到1*15的向量
    采用动态路由的方式
    """
    def __init__(self,
                 num_route,
                 num_capsule=10,                # equal to num_label?
                 in_channels=5,                 # num_capsule of primaryCaps
                 out_channels=14,               # output vector
                 num_iteration=3):              # iteration times of b_ij
        super(DigitCaps, self).__init__()

        self.in_channel = in_channels
        self.num_route = num_route
        self.num_capsule = num_capsule
        self.num_iteration = num_iteration
        self.W = nn.Parameter(torch.randn(1, num_route, num_capsule, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        # (B, features, 5) -> (B, features, 10, 5, 1)
        x = torch.stack([x]*self.num_capsule, dim=2).unsqueeze(4)
        # (1, features, 10, 14, 5) -> (B, features, 10, 14, 5)
        W = torch.cat([self.W]*batch_size, dim=0)
        # step1: 转换
        # u_hat: (B, features, 10, 14, 1)
        u_hat = torch.matmul(W, x)
        #　(1, features, 10, 1)
        b_ij = Variable(torch.zeros(1, self.num_route, self.num_capsule, 1))

        if config.use_cuda:
            b_ij = b_ij.cuda()

        for i in range(self.num_iteration):
            c_ij = F.softmax(b_ij)
            # (1, features, 10, 1) -> (B, features, 10, 1, 1)
            c_ij = torch.cat([c_ij]*batch_size, dim=0).unsqueeze(4)
            # (B, 1, 10, 14, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j)

            if i < self.num_iteration-1:
                # (B, features, 10, 1, 14) * (B, features, 10, 14, 1) = (B, features, 10, 1, 1)
                a_ij = torch.matmul(u_hat.transpose(3,4), torch.cat([v_j]*self.num_route, dim=1))
                # (1, features, 10, 1)
                b_ij  = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)
        # (B, 10, 14, 1)
        return v_j.squeeze(1)


def squash(tensor, dim=-1):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm)