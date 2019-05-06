import torch
import torch.utils.data as data
from torch.utils.data.sampler import RandomSampler


def get_collate_fn(pad_idx, fix_max_len=None):

    def _f(batch):
        tokens_li = []
        labels = []

        for ins in batch:
            tokens_li.append(ins['tokens'])
            try:
                labels.append(int(ins['label']))
            except ValueError:
                # When it's test data
                labels.append('')

        if not fix_max_len:
            max_len = max(len(tokens) for tokens in tokens_li)
        else:
            max_len = fix_max_len

        padded_tokens_li = []
        for tokens in tokens_li:
            if len(tokens) < max_len:
                # Too short
                padded_tokens = tokens + [pad_idx] * (max_len - len(tokens))
            else:
                # Too long
                padded_tokens = tokens[:max_len]
            padded_tokens_li.append(padded_tokens)

        return (
                torch.LongTensor(labels),
                torch.LongTensor(padded_tokens_li)
                )

    return _f


def get_dataloader(dataset, bsize, pad_idx, fix_max_len=None, shuffle=True):
    '''
    Build pytorch dataloder instance, given lf dataset.
    '''
    if shuffle:
        dataloader = data.DataLoader(
                dataset,
                batch_size=bsize,
                sampler=RandomSampler(dataset),
                collate_fn=get_collate_fn(pad_idx, fix_max_len)
                )
    else:
        dataloader = data.DataLoader(
                dataset,
                batch_size=bsize,
                shuffle=False,
                collate_fn=get_collate_fn(pad_idx, fix_max_len)
                )
    return dataloader
