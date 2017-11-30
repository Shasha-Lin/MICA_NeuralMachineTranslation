# Code source : https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

from data_loading import *
import torch

# Necessary peprocessing of data for modeling (details included in the Readme file)
def filterPair(p, min_length_input, min_length_target, max_length_input, max_length_target):
    return not (not (len(p[0].split(' ')) > min_length_input) or not (len(p[1].split(' ')) > min_length_target) or not (
        len(p[0].split(' ')) < max_length_input) or not (len(p[1].split(' ')) < max_length_target))


def filterPairs(pairs, min_length_input, min_length_target, max_length_input, max_length_target):
    return [pair for pair in pairs if filterPair(pair, min_length_input, min_length_target,
                                                 max_length_input, max_length_target)]

def prepare_data(lang1_name, lang2_name, min_length_input, max_length_input,
                 min_length_target, max_length_target, set_type='train', do_filter=True, normalize=False,
                 reverse=False, path='.', term='txt', char_output=False):

    # Get the source and target language class objects and the pairs (x_t, y_t)
    input_lang, output_lang, pairs = read_langs(lang1_name, 
                                                lang2_name, set_type=set_type,
                                                reverse=reverse,
                                                normalize=normalize,
                                                path=path, 
                                                term=term,
                                                char_output=char_output
                                               )
    print("Read %d sentence pairs" % len(pairs))

    if do_filter is True:
        pairs = filterPairs(pairs, min_length_input, min_length_target,
                            max_length_input, max_length_target)
        print("Filtered to %d pairs" % len(pairs))
    else:
        print("Pairs not filtered...")
    
    print("Indexing words...")
    if not char_output:
        for pair in pairs:
            input_lang.index_words(pair[0])
            output_lang.index_words(pair[1])
    else:
        for pair in pairs:
            input_lang.index_words(pair[0])
        output_lang.get_vocab(list(np.array(pairs)[:, 1]), from_filenames=False)
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    return input_lang, output_lang, pairs

def indexes_from_sentence(lang, sentence):
    return [lang.word2index(word) for word in sentence.split(' ')]


def variable_from_sentence(use_cuda, lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    input_lengths = [len(indexes)]
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        var = var.cuda()
    return var, input_lengths

            
def variables_from_pair(pair, input_lang, output_lang, char=False, use_cuda=False):
    input_variable = variable_from_sentence(input_lang, pair[0], use_cuda=use_cuda)
    target_variable = variable_from_sentence(output_lang, pair[1], char=char, use_cuda=use_cuda)
    return (input_variable, target_variable)

# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def random_batch(USE_CUDA, batch_size, pairs, input_lang, output_lang, max_length_input, max_length_output, char_output=False):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(input_lang, pair[0]))
        target_seqs.append(indexes_from_sentence(output_lang, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    
    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()
        
    return input_var, input_lengths, target_var, target_lengths
