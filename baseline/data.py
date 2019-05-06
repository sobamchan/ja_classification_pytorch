import pickle
from collections import Counter
from pathlib import Path

import fire
import lineflow as lf
from janome.tokenizer import Tokenizer


START_TOKEN = '<s>'
END_TOKEN = '</s>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


def get_preprocess(tokenizer):
    def _f(sample):
        data = sample['data'].replace('\n', '')
        tokens = tokenizer.tokenize(data, wakati=True)
        return {
                'tokens': tokens,
                'label': int(sample['label'])
                }
    return _f


def build_vocab(tokens, max_size=50000):
    counter = Counter(tokens)
    words, _ = zip(*counter.most_common(max_size))
    words = [PAD_TOKEN, UNK_TOKEN] + list(words)
    t2i = dict(zip(words, range(len(words))))
    if START_TOKEN not in t2i:
        t2i[START_TOKEN] = len(t2i)
        words += [START_TOKEN]
    if END_TOKEN not in t2i:
        t2i[END_TOKEN] = len(t2i)
        words += [END_TOKEN]

    return words, t2i


def get_postprocess(t2i, unk_index):
    def _f(sample):
        return {
                'tokens': [t2i.get(token, unk_index) for token in sample['tokens']],
                'label': sample['label']
                }
    return _f


def build(dpath, savedir):
    '''
    1. Read dpath csv file.
    2. Preprocess. (tokenizing, stripping)
    3. Build vocab.
    4. Replace tokens with ids.
    5. Save.
    '''

    # Load csv data.
    dpath = Path(dpath)
    savedir = Path(savedir)

    # Preprocess
    tokenizer = Tokenizer()
    train = lf.CsvDataset(str(dpath / 'train.csv'), header=True).map(get_preprocess(tokenizer))
    test = lf.CsvDataset(str(dpath / 'test.csv'), header=True).map(get_preprocess(tokenizer))

    # Collect all tokens.
    tokens = lf.flat_map(
            lambda x: x['tokens'],
            train,
            lazy=True
            )

    # Build vocab.
    words, t2i = build_vocab(tokens)

    # Save vocab.
    with open(savedir / 'vocab.pkl', 'wb') as f:
        pickle.dump((t2i, words), f)

    # Save dataset.
    train.map(get_postprocess(t2i, t2i[UNK_TOKEN])).save(str(savedir / 'dataset.train.token.pkl'))
    test.map(get_postprocess(t2i, t2i[UNK_TOKEN])).save(str(savedir / 'dataset.test.token.pkl'))


if __name__ == '__main__':
    fire.Fire()
