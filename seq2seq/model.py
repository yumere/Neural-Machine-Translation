import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import Vocabulary, reversed_basic_tokens
from configparser import ConfigParser
from mosestokenizer import MosesTokenizer

config = ConfigParser()
config.read(os.path.join(ROOT_DIR, "config.ini"))

default_config = {
    'epochs': config['DEFAULT'].getint('EPOCHS'),
    'gpu': config['DEFAULT'].get('GPU', None),
    'log_dir': config['DEFAULT'].get('LOG_DIR'),
    'save_dir': config['DEFAULT'].get('SAVE_DIR')
}

vocab_config = {
    'save': config['VOCABULARY'].getboolean('SAVE', fallback=True),
    'load': config['VOCABULARY'].getboolean('LOAD', fallback=False),
    'src_path': config['VOCABULARY'].get('SRC_PATH', fallback=None),
    'trg_path': config['VOCABULARY'].get('TRG_PATH', fallback=None),
    'vocab_size': config['VOCABULARY'].getint('SIZE')
}

model_config = {
    'lr': config['MODEL'].getfloat('LR'),
    'batch_size': config['MODEL'].getint('BATCH_SIZE'),
    'input_size': config['MODEL'].getint('INPUT_SIZE'),
    'hidden_size': config['MODEL'].getint('hidden_size'),
    'num_layers': config['MODEL'].getint('NUM_LAYERS')
}


class SequenceDataset(Dataset):
    def __init__(self, src_lines, src_vocab: Vocabulary, trg_lines, trg_vocab: Vocabulary):
        """

        :param src_lines:
        :param src_vocab:
        :param trg_lines:
        :param trg_vocab:
        """
        super(SequenceDataset, self).__init__()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.datasets = list(zip(src_lines, trg_lines))

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, item):
        src, trg = self.datasets[item]
        src_vec = self.src_vocab.to_vector(src)
        trg_vec = self.trg_vocab.to_vector(trg)

        src_vec += [self.src_vocab['<EOS>']]
        trg_vec += [self.trg_vocab['<EOS>']]

        return src_vec, len(src_vec), trg_vec, len(trg_vec)


def collate_fn(data):
    data.sort(key=lambda x: x[1], reverse=True)
    src, src_len, trg, trg_len = zip(*data)
    src_max_len, trg_max_len = max(src_len), max(trg_len)
    for s, t in zip(src, trg):
        s += [reversed_basic_tokens['<PAD>']] * (src_max_len - len(s))
        t += [reversed_basic_tokens['<PAD>']] * (trg_max_len - len(t))

    return src, src_len, trg, trg_len


class S2S(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, input_size, num_layers, hidden_size):
        super(S2S, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.src_embed = nn.Embedding(src_vocab_size, input_size, padding_idx=reversed_basic_tokens['<PAD>'])
        self.trg_embed = nn.Embedding(trg_vocab_size, input_size, padding_idx=reversed_basic_tokens['<PAD>'])

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, trg_vocab_size)

    def forward(self, *inputs):
        src, src_len, trg, trg_len, states = inputs
        src_embed = self.src_embed(src).transpose(0, 1)
        trg_embed = self.trg_embed(trg).transpose(0, 1)

        enc_output, states_t = self.encoder(src_embed, states)
        dec_output, states_t = self.decoder(trg_embed, states_t)
        dec_output = dec_output[:-1, :, :]

        flatten_output = dec_output.reshape(-1, self.hidden_size)
        output = self.linear(flatten_output)
        return output


def train():
    src_path = os.path.join(ROOT_DIR, "../datasets/30_length.en")
    trg_path = os.path.join(ROOT_DIR, "../datasets/30_length.fr")
    en_tokenizer = MosesTokenizer('en')
    fr_tokenizer = MosesTokenizer('fr')

    with open(src_path, "rt", encoding="utf8") as src, open(trg_path, "rt", encoding="utf8") as trg:
        src_lines = [en_tokenizer(line.strip()) for line in src.readlines()]
        trg_lines = [fr_tokenizer(line.strip()) for line in trg.readlines()]

    tqdm.write("Total sentences: {:,}".format(len(src_lines)))

    if vocab_config['load'] and vocab_config['src_path'] and vocab_config['trg_path']:
        src_vocab = Vocabulary.load(os.path.join(ROOT_DIR, vocab_config['src_path']))
        trg_vocab = Vocabulary.load(os.path.join(ROOT_DIR, vocab_config['trg_path']))
    else:
        src_vocab = Vocabulary.build_vocabulary(corpus=src_lines, max_vocab_size=vocab_config['vocab_size'])
        trg_vocab = Vocabulary.build_vocabulary(corpus=trg_lines, max_vocab_size=vocab_config['vocab_size'])

    if vocab_config['save']:
        src_vocab.save(os.path.join(ROOT_DIR, vocab_config['src_path']))
        trg_vocab.save(os.path.join(ROOT_DIR, vocab_config['trg_path']))

    datasets = SequenceDataset(src_lines, src_vocab, trg_lines, trg_vocab)
    dataloader = DataLoader(datasets, batch_size=model_config['batch_size'], shuffle=False, num_workers=1, collate_fn=collate_fn)

    model = S2S(src_vocab_size=vocab_config['vocab_size'], trg_vocab_size=vocab_config['vocab_size'],
                input_size=model_config['input_size'], num_layers=model_config['num_layers'],
                hidden_size=model_config['hidden_size']).cuda(device=default_config['gpu'])

    criterion = nn.CrossEntropyLoss(ignore_index=reversed_basic_tokens['<PAD>'], reduction='sum')
    optimizer = Adam(model.parameters(), lr=model_config['lr'])
    for epoch in tqdm(range(default_config['epochs']), desc='EPOCH', leave=True):
        with tqdm(total=len(src_lines), desc="Sent.", leave=False) as pbar:
            for step, (src_vec, src_len, trg_vec, trg_len) in enumerate(dataloader):
                states = [torch.zeros(model_config['num_layers'], len(src_vec), model_config['hidden_size']).cuda(device=default_config['gpu']) for i in range(2)]
                src_vec = torch.tensor(src_vec, dtype=torch.long).cuda(device=default_config['gpu'])

                trg_input = [[trg_vocab['<SOS>']] + vec for vec in trg_vec]
                trg_vec = torch.tensor(trg_vec, dtype=torch.long).cuda(device=default_config['gpu'])
                trg_input = torch.tensor(trg_input, dtype=torch.long).cuda(device=default_config['gpu'])

                pbar.update(model_config['batch_size'])
                optimizer.zero_grad()

                flatten_output = model(src_vec, src_len, trg_input, trg_len, states)
                flatten_trg = trg_vec.reshape(-1)

                loss = criterion(flatten_output, flatten_trg)
                loss /= src_vec.shape[0]
                loss.backward()

                optimizer.step()

                if step % 100 == 0:
                    tqdm.write("Step: {:,} Loss: {:,}".format(step, float(loss)))
        # torch.save(model.state_dict(), "{}_model.pt".format(epoch))

        # learning rate decay by 3 epoch
        lr = model_config['lr'] * (0.1 ** (epoch // 3))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def predict():
    src_path = os.path.join(ROOT_DIR, "../datasets/30_train.en")
    trg_path = os.path.join(ROOT_DIR, "../datasets/30_train.fr")

    with open(src_path, "rt") as src, open(trg_path, "rt") as trg:
        src_lines = [line.strip().split() for line in src.readlines()]
        trg_lines = [line.strip().split() for line in trg.readlines()]

    tqdm.write("Total sentences: {:,}".format(len(src_lines)))

    if vocab_config['load'] and vocab_config['src_path'] and vocab_config['trg_path']:
        src_vocab = Vocabulary.load(os.path.join(ROOT_DIR, vocab_config['src_path']))
        trg_vocab = Vocabulary.load(os.path.join(ROOT_DIR, vocab_config['trg_path']))
    else:
        src_vocab = Vocabulary.build_vocabulary(corpus=src_lines, max_vocab_size=vocab_config['vocab_size'])
        trg_vocab = Vocabulary.build_vocabulary(corpus=trg_lines, max_vocab_size=vocab_config['vocab_size'])

    datasets = SequenceDataset(src_lines, src_vocab, trg_lines, trg_vocab)
    dataloader = DataLoader(datasets, batch_size=model_config['batch_size'], shuffle=False, num_workers=4,
                            collate_fn=collate_fn)

    model = S2S(src_vocab_size=vocab_config['vocab_size'], trg_vocab_size=vocab_config['vocab_size'],
                input_size=model_config['input_size'], num_layers=model_config['num_layers'],
                hidden_size=model_config['hidden_size']).cuda(device=default_config['gpu'])
    with tqdm(total=len(src_lines), desc="Sent.", leave=False) as pbar:
        for step, (src_vec, src_len, trg_vec, trg_len) in enumerate(dataloader):
            src_vec, trg_vec = src_vec.cuda(device=default_config['gpu']), trg_vec.cuda(device=default_config['gpu'])
            pbar.update(model_config['batch_size'])
            flatten_output = model(src_vec, src_len, trg_vec, trg_len)
            logsoftmax_output = F.log_softmax(flatten_output, dim=1)
            output = logsoftmax_output.argmax(dim=1).cpu().numpy()
            output = np.split(output, model_config['batch_size'])


if __name__ == '__main__':
    train()
    # predict()
