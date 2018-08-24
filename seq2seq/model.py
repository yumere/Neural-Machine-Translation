import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import Vocabulary, reversed_basic_tokens
from configparser import ConfigParser
import argparse


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

        self._initialize_lstm()

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

    def _initialize_lstm(self):
        for name, params in self.encoder.named_parameters():
            if name.startswith("weight"):
                init.uniform_(params, -0.08, 0.08)


def train(args, config):
    device = torch.device("cuda:{}".format(args.cuda) if args.cuda else 'cpu')
    source_language = args.source_file.name.split(".")[-1]
    target_language = args.target_file.name.split(".")[-1]

    src_path = os.path.join(ROOT_DIR, "../datasets/30_length.en")
    trg_path = os.path.join(ROOT_DIR, "../datasets/30_length.fr")

    with open(src_path, "rt", encoding="utf8") as src, open(trg_path, "rt", encoding="utf8") as trg:
        src_lines = [line.strip().split(" ") for line in src.readlines()]
        trg_lines = [line.strip().split(" ") for line in trg.readlines()]

    tqdm.write("Total sentences: {:,}".format(len(src_lines)))

    # Load vocabulary
    if args.source_vocab and args.target_vocab:
        src_vocab = Vocabulary.load(args.source_vocab)
        trg_vocab = Vocabulary.load(args.target_vocab)
    # If not exist, build vocabulary from input files
    else:
        src_vocab = Vocabulary.build_vocabulary(corpus=src_lines, max_vocab_size=args.vocab_size, lang=source_language)
        trg_vocab = Vocabulary.build_vocabulary(corpus=trg_lines, max_vocab_size=args.vocab_size, lang=target_language)

        # If log_dir exists, save vocabulary
        if args.log_dir:
            src_vocab.save(os.path.join(ROOT_DIR, args.log_dir, "{}.{}".format(len(src_vocab), source_language)))
            trg_vocab.save(os.path.join(ROOT_DIR, args.log_dir, "{}.{}".format(len(trg_vocab), target_language)))

    datasets = SequenceDataset(src_lines, src_vocab, trg_lines, trg_vocab)
    dataloader = DataLoader(datasets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    model = S2S(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab),
                input_size=config['embedding_size'], num_layers=config['num_layers'],
                hidden_size=config['hidden_size']).cuda(device=device)

    criterion = nn.CrossEntropyLoss(ignore_index=reversed_basic_tokens['<PAD>'], reduction='sum')
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    for epoch in tqdm(range(args.epoch), desc='EPOCH', leave=True):
        with tqdm(total=len(src_lines), desc="Sent.", leave=False) as pbar:
            for step, (src_vec, src_len, trg_vec, trg_len) in enumerate(dataloader):
                states = [torch.zeros(config['num_layers'], len(src_vec), config['hidden_size']).cuda(device=device) for i in range(2)]
                src_vec = torch.tensor(src_vec, dtype=torch.long).cuda(device=device)

                trg_input = [[trg_vocab['<SOS>']] + vec for vec in trg_vec]
                trg_vec = torch.tensor(trg_vec, dtype=torch.long).cuda(device=device)
                trg_input = torch.tensor(trg_input, dtype=torch.long).cuda(device=device)

                pbar.update(args.batch_size)
                optimizer.zero_grad()

                flatten_output = model(src_vec, src_len, trg_input, trg_len, states)
                flatten_trg = trg_vec.reshape(-1)

                loss = criterion(flatten_output, flatten_trg)
                loss /= src_vec.shape[0]
                loss.backward()
                clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                if step % 100 == 0:
                    tqdm.write("Step: {:,} Loss: {:,}".format(step, float(loss)))

        # learning rate decay by epoch
        if epoch >= 5:
            lr = args.learning_rate * 0.5 ** (epoch-4)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_file', type=argparse.FileType('rt', encoding="UTF-8"), help="The source language file")
    parser.add_argument('target_file', type=argparse.FileType('rt', encoding="UTF-8"), help="The target language file")
    parser.add_argument('--source-vocab', type=str, help="source language vocabulary")
    parser.add_argument('--target-vocab', type=str, help="target language vocabulary")
    parser.add_argument('--log_dir', type=str, help="log directory")
    parser.add_argument('--cuda', type=int, default=None, help="GPU number (default: None)")

    parser.add_argument('--vocab_size', type=int, metavar='30000', default=30000, help="Vocabulary size")
    parser.add_argument('--num_workers', type=int, metavar='16', default=16, help="Num workers for dataloader")
    parser.add_argument('--batch-size', type=int, metavar='256', default=256, help="The number of mini batch")
    parser.add_argument('--epoch', type=int, metavar='10', default=10, help="The number of epochs")
    parser.add_argument("-lr", "--learning_rate", type=float, metavar='0.0002', default=0.0002, help="The learning rate")
    parser.add_argument("--decay", type=int, metavar='10', default=10, help="How often update learning rate")

    parser.add_argument("-c", "--config", type=str, required=True, help="The configuration file")
    parser.add_argument('-r', '--resume', type=str, default=None, help="path to the latest checkpoint (default: None)")

    args = parser.parse_args()
    config = ConfigParser()
    config.read(args.config)

    model_config = {
        'embedding_size': config['MODEL'].getint('embedding_size'),
        'hidden_size': config['MODEL'].getint('hidden_size'),
        'num_layers': config['MODEL'].getint('num_layers')
    }

    train(args, model_config)

