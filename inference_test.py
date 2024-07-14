from model import make_model
import torch
from utils import subsequent_mask


def inference_test():
    test_model = make_model(11, 11, 2)  # d_model=512, h=8, d_k=512/8=64
    test_model.eval()
    batch = 2
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)  # mask只会在query和key的对应计算后进行使用

    # 由于不进行训练直接进行推断，所以可以先计算出来自encoder的memory
    print("Encode")
    memory = test_model.encode(src, src_mask)
    # 如果输入的目标句子和原句子batch不等，也就是如果句子数不等，会导致广播机制异常而出错
    ys = torch.zeros(batch, 1).type_as(src)

    # 本质上也是进行greedy decode，所以掩码在这里实际上是没有用的
    print("Decode")
    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # 就是这里，虽然每次输出的长度都会变长，但是实际上预测的时候只会看最后一个，而注意力计算过程中也会关注之前生成的内容
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)  # _表示最大的数，第二个返回值表示在对应维度中最大的数的位置索引
        # print(out.shape, out[:, -1].shape, prob.shape, next_word.shape)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(batch, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


if __name__ == '__main__':
    run_tests()
