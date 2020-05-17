import torchtext
from torchtext.data.utils import get_tokenizer

# WikiText-2 has been proposed as a more realistic benchmark for language modeling
# than the pre-processed PennTreebank.
# WikiText-2 consists of around 2 million words extracted from Wikipedia articles.


def get_data(train_batch_size, eval_batch_size, device):
    """Build WikiText-2 dataset. Return a dictionary {"train_data": _, "val_data": _, "test_data": _}, and TEXT Field
    :param device: either cpu or gpu.
    :param train_batch_size:
    :param eval_batch_size:
    :return: input_dataset (dictionary); TEXT: (torchtext.data.Field)
    """
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)

    input_dataset = {
        "train_data": batchify(TEXT, train_txt, train_batch_size, device),
        "val_data": batchify(TEXT, val_txt, eval_batch_size, device),
        "test_data": batchify(TEXT, test_txt, eval_batch_size, device),
    }

    return input_dataset, TEXT


def batchify(TEXT, data, batch_size, device):
    """ Starting from sequential data, the batchify() arranges the dataset intp columns,
    trimming off any tokens remaining after the data has been divided into batches of size batch_size
    :param TEXT: torchtext.data.Field obj.
    :param data: output of torchtext.datasets.XXXXX.splits(TEXT)
    :param batch_size: batch_size, either train_batch_size or eval_batch_size
    :param device: string
    :return: processed data (to.(device))
    """
    data = TEXT.numericalize([data.examples[0].text])
    num_batches = data.size(0)//batch_size
    data = data.narrow(0, 0, num_batches * batch_size)
    data = data.view(batch_size, -1).t().contiguous()

    return data.to(device)
