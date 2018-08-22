from collections import Counter
from mosestokenizer import MosesTokenizer
from tqdm import tqdm

basic_tokens = {
    0: '<PAD>',
    1: '<UNK>',
    2: '<SOS>',
    3: '<EOS>'
}

reversed_basic_tokens = dict(zip(basic_tokens.values(), basic_tokens.keys()))


class Vocabulary(object):
    dictionary = basic_tokens
    reversed_dictionary = {}

    def __init__(self, lang='en', max_vocab_size=30000):
        self.lang = lang
        self.max_vocab_size = max_vocab_size

    def to_vector(self, words):
        vector = []
        for w in words:
            vector.append(self[w])
        return vector

    def to_string(self, vector_list, remove_pad=True):
        words = []
        for v in vector_list:
            v = int(v)
            if remove_pad and v == 0 or v == 1 or v == 3:
                continue
            words.append(self[v])

        return words

    @classmethod
    def build_vocabulary(cls, corpus: list=None, file_path: str=None, max_vocab_size=30000, lang='en'):
        vocab = cls(lang=lang, max_vocab_size=max_vocab_size)
        counter = Counter()

        tokenizer = MosesTokenizer(lang=lang)

        if file_path is not None:
            with open(file_path, "rt") as f:
                # TODO: Make preprocessor
                corpus = f.readlines()

        for sentence in tqdm(corpus, desc="Build vocabulary"):
            words = tokenizer(sentence.strip())
            counter.update(words)

        for index, (k, v) in enumerate(counter.most_common(max_vocab_size - len(basic_tokens))):
            vocab.dictionary[index + len(basic_tokens)] = k

        vocab.reversed_dictionary = dict(zip(vocab.dictionary.values(), vocab.dictionary.keys()))
        return vocab

    def save(self, file_path):
        with open(file_path, "wt") as f:
            for k, v in self.dictionary.items():
                # number, word
                f.write("{} {}\n".format(k, v))

    @classmethod
    def load(cls, file_path):
        vocab = cls()
        with open(file_path, "rt") as f:
            for line in f:
                number, word = line.split()
                vocab.dictionary.update({int(number): word})

        vocab.reversed_dictionary = dict(zip(vocab.dictionary.values(), vocab.dictionary.keys()))
        return vocab

    def __getitem__(self, item):
        if type(item) == int:
            return self.dictionary[item]
        elif type(item) == str:
            if item not in self.reversed_dictionary:
                return self.reversed_dictionary['<UNK>']  # return UNK token
            return self.reversed_dictionary[item]

    def __setitem__(self, key, value):
        if type(key) == int:
            self.dictionary[key] = value
        elif type(key) == str:
            self.reversed_dictionary[key] = value

    def __contains__(self, item):
        if type(item) == int:
            return self.dictionary.__contains__(item)
        elif type(item) == str:
            return self.reversed_dictionary.__contains__(item)

    def __len__(self):
        return len(self.dictionary)
