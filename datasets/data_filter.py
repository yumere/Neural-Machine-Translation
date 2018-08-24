import argparse
from multiprocessing import Pool

from mosestokenizer import MosesTokenizer
from tqdm import tqdm


def splitter(sentences, lang, max_tokens, max_length):
    results = []
    for sent in sentences:
        results.append(sent)
        if len(results) == max_length:
            yield results, max_tokens, lang
            results = []

    else:
        if len(results):
            yield results, max_tokens, lang
        else:
            raise StopIteration


def preprocess(inputs):
    sentences, max_tokens, lang = inputs
    tokenizer = MosesTokenizer(lang)

    result = []
    for sent in sentences:
        words = tokenizer(sent.strip())
        if len(words) > max_tokens:
            continue
        else:
            result.append(" ".join(words) + "\n")

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=argparse.FileType('rt', encoding='UTF-8'), help="The input file path")
    parser.add_argument('outfile', type=argparse.FileType('wt', encoding='UTF-8'), help="The output file path")
    parser.add_argument('lang', type=str, help='The language to be tokenized')
    parser.add_argument('max_tokens', type=int, help="Max tokens length")
    parser.add_argument('-c', '--max_cpu', type=int, default=10, help="The number of CPUs to use, default: 10")
    parser.add_argument('--mini_batch', type=int, default=100, help="Mini batch size for each CPU, default: 100")
    args = parser.parse_args()

    with args.infile as f:
        lines = f.readlines()

    loader = splitter(lines, args.lang, args.max_tokens, args.mini_batch)
    results = []
    with Pool(args.max_cpu) as p, tqdm(total=len(lines), desc="Sentences") as pbar:
        for sents in p.imap(preprocess, loader):
            results += sents
            pbar.update(len(sents))

    with args.outfile as f:
        f.writelines(results)
