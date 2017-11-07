# Code source : https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

from data_loading import *
import torch


# If GPU being used, set TRUE else FALSE:
USE_CUDA = torch.cuda.is_available()


# Necessary peprocessing of data for modeling (details included in the Readme file)

def prepare_data(lang1_name, lang2_name, reverse=False, path='.', term='txt', char_output=False):

    # Get the source and target language class objects and the pairs (x_t, y_t)
    input_lang, output_lang, pairs = read_langs(lang1_name, 
                                                lang2_name, 
                                                reverse=reverse, 
                                                path=path, 
                                                term=term,
                                                char_output=char_output
                                               )
    print("Read %d sentence pairs" % len(pairs))
    
    pairs = filter_pairs(pairs)
    print("Filtered to %d pairs" % len(pairs))
    
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


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence, char_output=False):
    if not char_output:
        return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    else:
        return [lang.word2index(word) for word in sentence.split(' ')] + [EOS_token]
        
# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def random_batch(batch_size, pairs, input_lang, output_lang, char_output=False):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(input_lang, pair[0]))
        target_seqs.append(indexes_from_sentence(output_lang, pair[1], char_output=char_output))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    
    # For input and target sequences, get array of lengths and pad with 0s to max length
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
