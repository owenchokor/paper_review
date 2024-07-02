import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_length, word_no, head_no):
        super(SelfAttention, self).__init__()
        