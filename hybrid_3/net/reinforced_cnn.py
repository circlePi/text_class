import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.attention import attention
from net.capsule import CapsNet
import config.config as config

os.environ["CUDA_VISIBLE_DEVICE"] = "%d"%config.device

class Reforced_CNN(nn.Module):
    def __init__(self,
                 word_embedding_dimension,
                 filters,
                 kernel_size,
                 attention_size):
        super(Reforced_CNN, self).__init__()
        self.attention_size = attention_size
        self.capsule = CapsNet(32, 112)  # (3,112)

        self.conv0 = nn.Conv1d(in_channels=word_embedding_dimension,
                               out_channels=filters,
                               kernel_size=kernel_size[0])

        self.conv1 = nn.Conv1d(in_channels=word_embedding_dimension,
                               out_channels=filters,
                               kernel_size=kernel_size[1])

        self.conv2 = nn.Conv1d(in_channels=word_embedding_dimension,
                               out_channels=filters,
                               kernel_size=kernel_size[2])

    def k_max_pooling(self, x, dim=2, k=3):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        x0 = self.conv0(inputs)
        x0 = F.relu(x0)
        x0 = self.k_max_pooling(x0)

        x1 = self.conv1(inputs)
        x1 = F.relu(x1)
        x1 = self.k_max_pooling(x1)

        x2 = self.conv2(inputs)
        x2 = F.relu(x2)
        x2 = self.k_max_pooling(x2)
        # (batch_size, 9, 32)
        x = torch.cat((x0, x1, x2), dim=2).permute(0, 2, 1)
        attention_output = attention(x, self.attention_size)   # (batch_size, 32)
        capsule_output = self.capsule(x)                       # (batch_size, num_capsule*out_channel)
        output = torch.cat((attention_output, capsule_output), dim=1)  # (batch_size, num_capsule*out_channel+32)
        return output                                                  # (batch_size, 172)