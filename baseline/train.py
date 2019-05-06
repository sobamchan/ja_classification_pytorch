import pickle
from pathlib import Path

import fire
from tqdm import tqdm
import numpy as np
import lineflow as lf
import torch
import torch.nn as nn
import torch.optim as optim

from baseline.dataloader import get_dataloader
from baseline.data import PAD_TOKEN
from baseline.models import Classifier


def run(dataset_dir, hid_n=128, emb_size=128, bsize=128, epoch=10, lr=0.01, use_cuda=False, seed=0):
    # For reproduction purpose, fix the random seed.
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset & vocab
    dataset_dir = Path(dataset_dir)

    train_dataset = lf.Dataset.load(dataset_dir / 'dataset.train.token.pkl')
    t2i, words = pickle.load(open(dataset_dir / 'vocab.pkl', 'rb'))

    # Select gpu or cpu
    device = torch.device('cuda' if use_cuda else 'cpu')

    voc_n = len(t2i)
    pad_idx = t2i[PAD_TOKEN]

    # Limit the lengths of input sentences.
    fix_max_len = 50

    # Get dataloader
    train_dataloader = get_dataloader(train_dataset, bsize, pad_idx, fix_max_len=fix_max_len)

    # Initialize model
    model = Classifier(voc_n, pad_idx, hid_n, emb_size, dropout=0)
    model = model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss function for classification task.
    loss_func = nn.CrossEntropyLoss()

    model.train()
    for i_epoch in tqdm(range(1, epoch + 1), total=epoch):
        losses = []
        for labels, inputs in train_dataloader:
            labels = labels.to(device)
            inputs = inputs.to(device)

            hid = model.init_hidden(inputs)
            hid = hid.to(device)

            out = model(inputs, hid)  # [B, C]
            loss = loss_func(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        tqdm.write(str(np.mean(losses)))

    test_dataset = lf.Dataset.load(dataset_dir / 'dataset.test.token.pkl')
    test_dataloader = get_dataloader(test_dataset, bsize, pad_idx, shuffle=False, fix_max_len=fix_max_len)
    pred_labels = []

    model.eval()
    with torch.no_grad():
        for _, inputs in test_dataloader:
            inputs = inputs.to(device)
            hid = model.init_hidden(inputs)
            hid = hid.to(device)

            out = model(inputs, hid)
            pred_labels += map(str, out.argmax(dim=1).tolist())

    with open('pred.txt', 'w') as f:
        f.write('\n'.join(pred_labels))


if __name__ == '__main__':
    fire.Fire()
