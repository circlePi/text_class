import torch
import torch.nn as nn

from net.attention import attention
from net.capsule import CapsNet


class Reinforced_GRU(nn.Module):
    def __init__(self,
                 word_embedding_dimension,
                 hidden_size,
                 bi_flag,
                 num_layer,
                 attention_size):
        super(Reinforced_GRU, self).__init__()
        self.rnn_cell = nn.GRU(input_size=word_embedding_dimension,
                               hidden_size=hidden_size,
                               num_layers=num_layer,
                               batch_first=True,
                               bidirectional=bi_flag)
        self.capsule = CapsNet(256, 23*16)
        self.attention_size = attention_size

    def forward(self, inputs):
        gru_output, _ = self.rnn_cell(inputs) # (batch_size, time_steps, 256)
        attention_output = attention(gru_output, self.attention_size) # (batch_size, 256)
        capsule_output = self.capsule(gru_output)                     # (batch_size, num_capsule*out_channel)
        output = torch.cat((attention_output, capsule_output), dim=1) # (batch_size, num_capsule*out_channel+256)
        return output                                                 # (batch_size, 396)




