import os

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

from utils import Vocabulary, reversed_basic_tokens
from tqdm import tqdm


GPU = None or torch.device('cuda:2')
LR = 0.05
BATCH_SIZE = 128


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
        trg_vec += [self.src_vocab['<EOS>']]

        return src_vec, len(src_vec), trg_vec, len(trg_vec)


def collate_fn(data):
    data.sort(key=lambda x: x[1], reverse=True)
    src, src_len, trg, trg_len = zip(*data)
    max_len = [max(src_len), max(trg_len)]
    for s, t in zip(src, trg):
        s += [reversed_basic_tokens['<PAD>']] * (max_len[0] - len(s))
        t += [reversed_basic_tokens['<PAD>']] * (max_len[1] - len(t))

    return torch.tensor(src, dtype=torch.long), torch.tensor(src_len, dtype=torch.long), \
           torch.tensor(trg, dtype=torch.long), torch.tensor(trg_len, dtype=torch.long)


class S2S(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, input_size, num_layers, hidden_size):
        super(S2S, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.src_embed = nn.Embedding(src_vocab_size, input_size, padding_idx=reversed_basic_tokens['<PAD>'])
        self.trg_embed = nn.Embedding(trg_vocab_size, hidden_size, padding_idx=reversed_basic_tokens['<PAD>'])
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, trg_vocab_size)

    def forward(self, *inputs):
        src, src_len, trg, trg_len, *_ = inputs
        sos_tokens = torch.full((trg.shape[0], 1), reversed_basic_tokens['<SOS>'], device=GPU)
        trg = torch.cat([sos_tokens.type(torch.float32), trg.type(torch.float32)], dim=1).type(torch.long).cuda(device=GPU)
        src_embed = self.src_embed(src).transpose(0, 1)
        trg_embed = self.trg_embed(trg).transpose(0, 1)

        # h_0, c_0 shape: (num_layers * num_directions, batch, hidden_size)
        enc_h_0, enc_c_0 = [torch.zeros(self.num_layers, src_embed.shape[1], self.hidden_size, dtype=torch.float32, device=GPU) for i in range(2)]
        enc_output, (enc_h_t, enc_c_t) = self.encoder(src_embed, (enc_h_0, enc_c_0))

        dec_h_0 = enc_h_t
        dec_c_0 = torch.zeros(self.num_layers, src_embed.shape[1], self.hidden_size, dtype=torch.float32, device=GPU)
        dec_output, (dec_h_t, dec_c_t) = self.decoder(trg_embed, (dec_h_0, dec_c_0))
        # dec_output.shape: seq_len, batch_size, hidden_size
        dec_output = dec_output[:-1, :, :]
        flatten_output = dec_output.reshape(-1, self.hidden_size)
        logits = self.linear(flatten_output)
        return logits


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(ROOT_DIR, "../datasets/30_length.en")
    trg_path = os.path.join(ROOT_DIR, "../datasets/30_length.fr")

    with open(src_path, "rt") as src, open(trg_path, "rt") as trg:
        src_lines = [line.strip().split() for line in src.readlines()]
        trg_lines = [line.strip().split() for line in trg.readlines()]

    tqdm.write("Total sentences: {:,}".format(len(src_lines)))

    src_vocab = Vocabulary.build_vocabulary(corpus=src_lines, max_vocab_size=30000)
    trg_vocab = Vocabulary.build_vocabulary(corpus=trg_lines, max_vocab_size=30000)

    datasets = SequenceDataset(src_lines, src_vocab, trg_lines, trg_vocab)
    dataloader = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = S2S(src_vocab_size=30000, trg_vocab_size=30000, input_size=300, num_layers=1, hidden_size=300).cuda(device=GPU)

    nllloss = nn.NLLLoss(size_average=False, ignore_index=reversed_basic_tokens['<PAD>'])
    optimizer = Adam(model.parameters(), lr=LR)
    with tqdm(total=len(src_lines), desc="Sent.") as pbar:
        for step, (src_vec, src_len, trg_vec, trg_len) in enumerate(dataloader):
            src_vec, trg_vec = src_vec.cuda(device=GPU), trg_vec.cuda(device=GPU)

            pbar.update(BATCH_SIZE)
            optimizer.zero_grad()

            flatten_output = model(src_vec, src_len, trg_vec, trg_len)
            logsoftmax_output = F.log_softmax(flatten_output, dim=1)
            flatten_trg = trg_vec.reshape(-1)

            loss = nllloss(logsoftmax_output, flatten_trg)
            loss /= src_vec.shape[0]

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                tqdm.write("Step: {:,} Loss: {:,}".format(step, float(loss)))
