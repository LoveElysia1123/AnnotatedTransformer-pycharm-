from model import make_model, DummyOptimizer, DummyScheduler
from trainer import *
from utils import *

warnings.filterwarnings("ignore")


# Load spacy tokenizer models, download them if they haven't been
# downloaded already
def load_tokenizers():
    """
    加载spaCy的德语和英语分词器模型
    """
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    """
    使用给定的分词器将句子分割成单词列表
    :param text: 需要进行分词的句子
    :param tokenizer: 需要使用的分词器
    :return: 分词后的单词列表
    """
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    """
    生成器函数，接受一个数据迭代器，每次抛出一个分词后的单词列表
    :param data_iter: 给定的句子迭代器
    :param tokenizer: 使用的分词器
    :param index: 句子元组中需要进行分词的句子索引
    :return: 按照迭代器顺序依次抛出每个句子的分词结果列表
    """
    for from_to_tuple in data_iter:  # 从数据迭代器中获取相对应的句子元组
        yield tokenizer(from_to_tuple[index])  # 从句子元组中获取指定index的句子进行分词并抛出分词后的单词列表


def build_vocabulary(spacy_de, spacy_en):
    """
    使用输入的分词器模型对所有数据集进行分词，返回数据集构建的vocab
    :param spacy_de: 德语分词器模型
    :param spacy_en: 英语分词器模型
    :return: 德语vocab和英语vocab
    """

    def tokenize_de(text):  # 调用德语分词器对输入的文本进行分词
        return tokenize(text, spacy_de)

    def tokenize_en(text):  # 调用英语分词器对输入的文本进行分词
        return tokenize(text, spacy_en)

    datasets.multi30k.URL[
        "train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    datasets.multi30k.URL[
        "valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
    datasets.multi30k.URL[
        "test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz"

    datasets.multi30k.MD5["train"] = "20140d013d05dd9a72dfde46478663ba05737ce983f478f960c1123c6671be5e"
    datasets.multi30k.MD5["valid"] = "a7aa20e9ebd5ba5adce7909498b94410996040857154dab029851af3a866da8c"
    datasets.multi30k.MD5["test"] = "6d1ca1dba99e2c5dd54cae1226ff11c2551e6ce63527ebb072a1f70f72a5cd36"

    print("Building German Vocabulary ...")
    # 默认返回训练集、验证集和测试集(均为迭代器)，其中language_pair的顺序是(源语言, 目标语言)
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),  # 索引0的位置即为源语言句子，该参数仅接受迭代器
        min_freq=2,  # 构建语料库的最低出现频率
        specials=["<s>", "</s>", "<blank>", "<unk>"],  # 特殊单词标记
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    # 设置未出现词的索引
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    """
    加载vocab，如果本地有就直接读取本地的，如果没有就调用build_vocabulary构建vocab并保存到本地
    :param spacy_de: 德语分词器模型
    :param spacy_en: 英语分词器模型
    :return: 德语vocab和英语vocab
    """
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


def collate_batch(
        batch,
        src_pipeline,
        tgt_pipeline,
        src_vocab,
        tgt_vocab,
        device,
        max_padding=128,
        pad_id=2,
):
    """
    将数据集迭代器输入后，使用对应的分词器对句子进行分词操作后填充到最大长度
    :param batch: 需要处理的数据集迭代器，每次抛出的数据应该为(源语言句子, 目标语言句子)
    :param src_pipeline: 源语言分词器
    :param tgt_pipeline: 目标语言分词器
    :param src_vocab: 源语言vocab
    :param tgt_vocab: 目标语言vocab
    :param device: 后续模型调用的设备
    :param max_padding: 最大填充长度
    :param pad_id: 填充标记id
    :return: 经过分词、token映射和填充处理的源语言数据集和目标语言数据集(src, tgt)
    """
    bs_id = torch.tensor([0], device=device)  # <s> token id, 句子开始标记
    eos_id = torch.tensor([1], device=device)  # </s> token id, 句子结束标记
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                # src_pipeline实际上是分词器模型，这里实际上将传进的句子进行分词操作后接上句子的开始和结束标记
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            dim=0,  # 在第一个维度上进行拼接
        )
        # 对目标句子使用相同的操作，唯一的不同在于使用的分词器不同
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (0,  # 向左填充的数量为0
                 max_padding - len(processed_src),  # 向右填充到最大长度
                 ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=12000,
        max_padding=128,
        is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch=batch,
            src_pipeline=tokenize_de,
            tgt_pipeline=tokenize_en,
            src_vocab=vocab_src,
            tgt_vocab=vocab_tgt,
            device=device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    # 获取训练集、验证集和测试集的数据迭代器
    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    # 将迭代器类型的数据集转换为可以使用dataset[index]形式获取的数据
    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None  # 用于多卡训练
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


def train_worker(
        gpu,
        ngpus_per_node,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        config,
        is_distributed=False,
):
    device = torch.device("cuda:" + str(gpu-1) if torch.cuda.is_available() else "cpu")
    print(f"Train worker process using device: {device} for training", flush=True)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]  # 使用空白填充符进行
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    if torch.cuda.is_available():
        model.cuda(gpu)
    else:
        model.to(device)

    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    if torch.cuda.is_available():
        criterion.cuda(gpu)
    else:
        criterion.to(device)

    train_dataloader, valid_dataloader = create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        train_worker(
            -1, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):  # 如果模型不存在，则立刻进行训练并保存模型
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model


if __name__ == '__main__':
    # 语料库加载
    spacy_de, spacy_en = load_tokenizers()  # 获取分词器模型
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)  # 使用分词器模型处理数据集得到语料库
    # 直接进行训练
    model = load_trained_model()
