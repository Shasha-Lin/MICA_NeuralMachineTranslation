# Code source : https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

# The process for preparing the data is:
#   - Read text file and split into lines
#   - Split lines into pairs and normalize
#   - Filter to pairs of a certain length
#   - Make word lists from sentences in pairs


import re
import socket
import torch
from masked_cross_entropy import *
from language_objects import CharTokenizer, Lang
import numpy as np
import random

#We'll need a unique index per word to use as the inputs and targets of the networks later. 
#To keep track of all this we will use a helper class called Lang which has word → index (word2index) and index → word (index2word) dictionaries, as well as a count of each word (word2count). 
#This class includes a function trim(min_count) to remove rare words once they are all counted.

PAD_token = 0
SOS_token = 1
EOS_token = 2

            
def normalize_string(s):
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def read_langs(lang1, lang2, set_type="train", normalize=False, path='.',
               term="txt", reverse=False, char_output=False):
    print("Reading lines...")
    # Read the file and split into lines
    # Attach the path here for the source and target language dataset
    if set_type == "train":
        filename = '%s/train/%s-%s.%s' % (path, lang1, lang2, term)
    elif set_type == "dev":
        filename = '%s/dev/%s-%s.%s' % (path, lang1, lang2, term)
    elif set_type == "valid":
        filename = '%s/dev/%s-%s.%s' % (path, lang1, lang2, term)
    elif set_type == "tst2010":
        filename = '%s/test/%s-%s.tst2010-%s' % (path, lang1, lang2, term)
    elif set_type == "tst2011":
        filename = '%s/test/%s-%s.tst2011-%s' % (path, lang1, lang2, term)
    elif set_type == "tst2012":
        filename = '%s/test/%s-%s.tst2012-%s' % (path, lang1, lang2, term)
    elif set_type == "tst2013":
        filename = '%s/test/%s-%s.tst2013-%s' % (path, lang1, lang2, term)
    elif set_type == "tst2014":
        filename = '%s/test/%s-%s.tst2014-%s' % (path, lang1, lang2, term)
    else:
        raise ValueError("set_type not found. Check data folder options")

    # lines contains the data in form of a list
    lines = open(filename).read().strip().split('\n')
    # Split every line into pairs
    if normalize == True:
        pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    else:
        pairs = [[s for s in l.split('\t')] for l in lines]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        if char_output:
            output_lang = CharTokenizer(vocab_file='')
        else:
            output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

