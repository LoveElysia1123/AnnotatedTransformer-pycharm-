import warnings
warnings.filterwarnings("ignore")
from trainer import *
from model import make_model, DummyOptimizer, DummyScheduler
from utils import *


def data_gen(V, batch_size, nbatches, device):
    """
    随机生成src和tgt一样的句子对，编码范围是1到V（不包含V），另外将0设置为填充标记，总共存在的单词数就是V
    :param V: 用于简单句子生成的语料库大小
    :param batch_size: 每个迭代器中包含的句子个数
    :param nbatches: 数据生成迭代器的迭代次数
    :param device: 数据的设备位置
    """
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src.to(device), tgt.to(device), 0)  # 使用yield时代表该函数是一个generator，返回的是迭代器





def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    基于输入的句子进行encode，并使用decoder生成一个新的目标句子（每次调用只能生成一个句子）
    :param model: 进行前向传播的transformer模型
    :param src: 原句子
    :param src_mask:  原句子掩码
    :param max_len: 最大句子长度，模型在decode到这个长度前将一直运行
    :param start_symbol: 句子开始标记
    :return: 对原句子进行encode后，使用decoder生成的句子
    """
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])  # 每次选取预测的最后一个位置作为新单词的概率
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


# Train the simple copy task.


def example_simple_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    V = 11  # 词汇表单词个数是V，包含填充标记0
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)  # smoothing=0就是相当于不使用smoothing
    model = make_model(V, V, N=2).to(device)  # model的词汇表个数是V（包含填充标记）
    print('Model Device: ', next(model.parameters()).device)

    # 设置优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    # 学习率调整
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_iter=data_gen(V, batch_size, 20, device),  # 每轮运行的时候再生成一个batch的数据输入进去
            model=model,
            loss_compute=SimpleLossCompute(model.generator, criterion),
            optimizer=optimizer,
            scheduler=lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5, device),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(device)
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len).to(device)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # for iter in data_gen(11, 80, 20, device):
    #     print(iter.src_mask.shape, iter.tgt_mask.shape)
    example_simple_model()
