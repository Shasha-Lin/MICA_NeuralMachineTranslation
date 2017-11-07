# https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/tools/tokenizer.py

from __future__ import unicode_literals
import os
import string
import codecs
import logging
import sys
from collections import Counter
import torch
#from .config import *

#sys.path.append(os.path.abspath(os.path.join(
#    os.path.dirname(__file__), './subword-nmt')))
#import learn_bpe
#import apply_bpe
PAD_TOKEN = 0
SOS_TOKEN = BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

class Lang(object):
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True
        
        keep_words = []
        
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens

        for word in keep_words:
            self.index_word(word)


class Tokenizer(Lang):

    def __init__(self, max_length=500, vocab_file=None,
                 additional_tokens=None,
                 vocab_threshold=2):
        self.max_length = max_length
        self.vocab_threshold = vocab_threshold
        self.special_tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        if additional_tokens is not None:
            self.special_tokens += additional_tokens
        self.__word2idx = {}
        if os.path.isfile(vocab_file):
            self.load_vocab(vocab_file)
        self.n_words = 3
        self.trimmed = False

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)

    def idx2word(self, idx):
        if idx < len(self.special_tokens):
            return self.special_tokens[idx]
        else:
            return self.vocab[idx - len(self.special_tokens)][0]

    def update_word2idx(self):
        self.__word2idx = {
            word[0]: idx + len(self.special_tokens) for idx, word in enumerate(self.vocab)}
        for i, tok in enumerate(self.special_tokens):
            self.__word2idx[tok] = i
        self.n_words=self.vocab_size

    def word2idx(self, word):
        return self.__word2idx.get(word, UNK_TOKEN)
    
    def word2index(self, word): 
        return self.__word2idx.get(word, UNK_TOKEN)
    
    def segment(self, line):
        """segments a line to tokenizable items"""
        return str(line).lower().translate(string.punctuation).strip().split()

    def get_vocab(self,  item_list, from_filenames=True, limit=None):
        vocab = Counter()
        if from_filenames:
            filenames = item_list
            # get combined vocabulary of all input files
            for fname in filenames:
                with codecs.open(fname, encoding='UTF-8') as f:
                    for line in f:
                        for word in self.segment(line):
                            vocab[word] += 1
        else:
            for line in item_list:
                for word in self.segment(line):
                    vocab[word] += 1
        self.vocab = vocab.most_common(limit)
        self.update_word2idx()

    def save_vocab(self, vocab_filename):
        if self.vocab is not None:
            with codecs.open(vocab_filename, 'w', encoding='UTF-8') as f:
                for (key, freq) in self.vocab:
                    f.write("{0} {1}\n".format(key, freq))

    def load_vocab(self, vocab_filename, limit=None):
        vocab = Counter()
        with codecs.open(vocab_filename, encoding='UTF-8') as f:
            for line in f:
                word, count = line.strip().split()
                vocab[word] = int(count)
        self.vocab = vocab.most_common(limit)
        self.update_word2idx()

    def tokenize(self, line, insert_start=None, insert_end=None):
        """tokenize a line, insert_start and insert_end are lists of tokens"""
        inputs = self.segment(line)
        targets = []
        if insert_start is not None:
            targets += insert_start
        for w in inputs:
            targets.append(self.word2idx(w))
        if insert_end is not None:
            targets += insert_end
        return torch.LongTensor(targets)

    def detokenize(self, inputs, delimiter=u' '):
        return delimiter.join([self.idx2word(idx) for idx in inputs]).encode('utf-8')
        # Remove words below a certain count threshold
    
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        no_words_kept = 0
        for (k,v) in self.vocab:
            if v >= min_count:
                keep_words.append([k]*v)
                no_words_kept += 1
        keep_words = [item for sublist in keep_words for item in sublist]
        print('keep_words %s / %s = %.4f' % (
            no_words_kept, len(self.vocab), no_words_kept / len(self.vocab)
        ))

        # Reinitialize dictionaries
        self.vocab = {}
        self.__word2idx = {}
        self.n_words = 3 # Count default tokens

        self.get_vocab(keep_words, from_filenames=False)

    
    
    
            
# adding section for character encoding on non-bpe base files
# Source: https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/tools/tokenizer.py
class CharTokenizer(Tokenizer):

    def segment(self, line):
        return list(line.strip())

    def detokenize(self, inputs, delimiter=u''):
        return super(CharTokenizer, self).detokenize(inputs, delimiter)

                