# Code source : https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

import torch
import torch.nn as nn
from torch.autograd import Variable
import io
#from data_loader import *
import matplotlib.pyplot as plt
import visdom
vis = visdom.Visdom()
import re
from nltk.translate import bleu_score
import random

from data_for_modeling import  EOS_token, SOS_token, pad_seq


# Evaluation is mostly the same as training, but there are no targets. Instead we always feed the decoder's predictions back to itself. 
# Every time it predicts a word, we add it to the output string. If it predicts the EOS token we stop there. We also store the decoder's attention outputs for each step to display later.

def indexes_from_sentence(lang, sentence, char=False):
    if char:
        return [lang.word2index(word) for word in sentence.split(' ')]
    else:
        return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(use_cuda, lang, sentence, max_length, char=False):

    #input_lengths = [len(sentence)]
    input_lengths = [max_length]
    
    indexes = indexes_from_sentence(lang, sentence, char)
    indexes.append(EOS_token)
    #indexes = [indexes]
    input_padded = [pad_seq(indexes, max_length)]
    input_batches = Variable(torch.LongTensor(input_padded), volatile=True).transpose(0, 1)
    if use_cuda:
        input_batches = input_batches.cuda()
    return input_batches, input_lengths

def evaluate_old(use_cuda, in_lang, out_lang, encoder, decoder,
             sentence, max_length):
    input_batch, input_lengths = variable_from_sentence(use_cuda, in_lang, sentence)
        
    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)
    
    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True) # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    
    if use_cuda:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        words = re.sub(' <EOS>', '', words)
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if use_cuda: decoder_input = decoder_input.cuda()
    # Set back to training mode
    encoder.train(True)
    decoder.train(True)
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]





def update_dictionary(target_sequence, topv, topi, key, dec_hidden, decoder_attns):
    if len(target_sequence) == 0:
        for i in range(len(topi)):
            target_sequence.update({str(topi[i]) : [topv[i], dec_hidden, decoder_attns] })
    else:
        prev_val = target_sequence[key][0]
        for i in range(len(topi)):
            target_sequence.update({key+"-"+str(topi[i]) : [topv[i]*prev_val, dec_hidden, decoder_attns] })
        del[target_sequence[key]]



def get_seq_through_beam_search(max_length, decoder, decoder_input, decoder_hidden, decoder_attentions, encoder_outputs, kmax, out_lang, use_cuda=False, char=False):
    target_sequence = dict()
    
    # Run through decoder
    for di in range(max_length):
        
        if di == 0:
            decoder_output, decoder_hidden, decoder_attention = decoder( decoder_input, decoder_hidden, encoder_outputs )
            topv, topi = decoder_output.data.topk(kmax)
            topv = topv[0]
            topi = topi[0]
            decoder_attentions[di,:decoder_attention.size(1)] += decoder_attention.data
            update_dictionary(target_sequence, topv, topi, None, decoder_hidden, decoder_attentions)
        else:
            temp = target_sequence.copy()
            keys = list(temp.keys())
            for i in range(len(keys)):
                inp = int(keys[i].split("-")[-1] if len(keys[i]) > 1 else keys[i])
                if inp != EOS_token:
                    dec_input = Variable(torch.LongTensor([inp]))
                    dec_input = dec_input.cuda() if use_cuda else dec_input
                    decoder_output, dec_hidden, decoder_attention = decoder( dec_input, temp[keys[i]][1],  encoder_outputs)
                    topv, topi = decoder_output.data.topk(kmax)
                    topv = topv[0]
                    topi = topi[0]
                    dec_attns = temp[keys[i]][2]
                    dec_attns[di,:decoder_attention.size(1)] += decoder_attention.data
                    update_dictionary(target_sequence, topv, topi, keys[i], dec_hidden, dec_attns)
        
        # Sort the target_Sequence dictionary to keep top k sequences only
        target_sequence = dict(sorted(target_sequence.items(), key=lambda x: x[1][0], reverse=True)[:kmax])
     
    # Get the sequence, decoder_attentions with maximum probability
    pair = sorted(target_sequence.items(), key=lambda x: x[1][0], reverse=True)[:1][0]
    seq = pair[0]
    decoder_attentions = pair[1][2]
    
    # Get the decoded words:
    decoded_words_indices = seq.split("-")
    if char:
        # decoded_words = [str(out_lang.idx2word(int(i))) for i in decoded_words_indices]
        decoded_words = [out_lang.index2word[int(i)] for i in decoded_words_indices]

    else:
        decoded_words = [out_lang.index2word[int(i)] for i in decoded_words_indices]
    if int(decoded_words_indices[-1]) != EOS_token:
        decoded_words.append('<EOS>')
    
    return decoded_words, decoder_attentions

### eduardo's evaluate function
def evaluate(input_lang, output_lang, encoder, decoder, sentence, max_length, kmax,use_cuda=False, char=False):
    """
    Function that generate translation.
    First, feed the source sentence into the encoder and obtain the hidden states from encoder.
    Secondly, feed the hidden states into the decoder and unfold the outputs from the decoder.
    Lastly, for each outputs from the decoder, collect the corresponding words in the target language's vocabulary.
    And collect the attention for each output words.
    @param encoder: the encoder network
    @param decoder: the decoder network
    @param sentence: string, a sentence in source language to be translated
    @param max_length: the max # of words that the decoder can return
    
    @output decoded_words: a list of words in target language
    @output decoder_attentions: a list of vector, each of which sums up to 1.0
    """
    
    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)    

    # process input sentence
    #input_variable = variable_from_sentence(use_cuda, input_lang, sentence)
    #input_length = input_variable[0].size()[0]
    
    input_batch, input_lengths = variable_from_sentence(use_cuda, input_lang, sentence, max_length)
        
    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batch, input_lengths, None)

    
    # encode the source lanugage
    #encoder_hidden = encoder.initHidden()
    #encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    #for ei in range(input_lengths):
    #    encoder_output, encoder_hidden = encoder(input_variable[ei],
    #                                             encoder_hidden)
    #    encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
    
    # decode the context vector
    decoder_hidden = encoder_hidden # decoder starts from the last encoding sentence
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    # output of this function
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    decoder_attentions = decoder_attentions.cuda() if use_cuda else decoder_attentions
    
    decoded_words, decoder_attentions = get_seq_through_beam_search(max_length, decoder, decoder_input, decoder_hidden, 
                                                                    decoder_attentions, encoder_outputs, kmax, output_lang, use_cuda=use_cuda, char=char)
    
    # Set back to training mode
    encoder.train(True)
    decoder.train(True)    
    
    return decoded_words, decoder_attentions[:len(decoded_words)+1, :len(encoder_outputs)]










# We can evaluate random sentences from the training set and print out the input,
# target, and output to make some subjective quality judgements:
def evaluate_randomly(use_cuda, in_lang, out_lang, encoder, decoder, input_sentence,
                      max_length, target_sentence=None):
    [input_sentence, target_sentence] = random.choice(pairs)
    evaluate_and_show_attention(use_cuda, in_lang, out_lang, encoder, decoder, input_sentence,
                      max_length, target_sentence)

def show_plot_visdom():
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    attn_win = 'attention (%s)' % hostname
    vis.image(torchvision.transforms.ToTensor()(Image.open(buf)), win=attn_win, opts={'title': attn_win})

def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    show_plot_visdom()
    plt.show()
    plt.close()

def evaluate_and_show_attention(use_cuda, in_lang, out_lang, encoder, decoder,
                                input_sentence, max_length, target_sentence=None):
    output_words, attentions = evaluate(use_cuda, in_lang, out_lang, encoder, decoder,
                                        input_sentence, max_length)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)
    show_attention(input_sentence, output_words, attentions)
    
    # Show input, target, output text in visdom
    win = 'evaluted (%s)' % hostname
    text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sentence, target_sentence, output_sentence)
    vis.text(text, win=win, opts={'title': win})
    
    
    
    
    
# eduardo's evaluate functions
def evaluateRandomly(input_lang, output_lang, encoder, decoder, max_length,kmax, pairs, n=5, char=False):
    """
    Randomly select a English sentence from the dataset and try to produce its French translation.
    Note that you need a correct implementation of evaluate() in order to make this function work.
    """
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, pair[0], max_length, kmax=kmax, char=char)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def eval_single(input_lang, output_lang, encoder, decoder, string, max_length,kmax, char=False):
    
    words, tensor = evaluate(input_lang, output_lang, encoder, decoder, string, max_length=max_length, kmax=kmax, char=char)
    words = ' '.join(words)
    words = re.sub(' <EOS>', '', words)
    return(words)
    
def evaluate_dev(input_lang, output_lang, encoder, decoder, list_strings, 
                 max_length, kmax, char=False):
    
    output = [eval_single(input_lang, output_lang, encoder, decoder, x[0], max_length,kmax, char=char) for x in list_strings]
    
    return(output)
