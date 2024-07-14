import copy
import math
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


def clones(module, N):
    "生成N个相同的层。"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """掩盖后续位置"""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    # 在inference test里面，就是简单的使用广播机制广播到每个单词与原句子计算的注意力矩阵里面，
    # 假如原句子的长度是n，当前生成的长度是m，这时注意力权重矩阵的维度是[m, n]那么广播机制就会从1行广播到m行来进行mask
    # 在simple copy task里面也是同样的道理，只不过这一步实现在了Batch类里面
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print("Attention", scores.shape, mask.shape)
    if mask is not None:
        # 如果是[2, 8, 10, 10]和[1, 1, 1, 10]的scores和mask, 也就是特定单词的mask，会被广播为[2, 8, 10, 10]
        scores = scores.masked_fill(mask == 0, -1e9)  # 在符合条件的位置使用第二个参数进行填充
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )