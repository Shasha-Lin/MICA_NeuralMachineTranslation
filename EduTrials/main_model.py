# -*- coding: utf-8 -*-

import random
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from comet_ml import Experiment

"""
Created on Wed Nov  8 22:31:02 2017

@author: eduardofierro
"""

######## File params ########

parser = argparse.ArgumentParser()
parser.add_argument('--MIN_LENGTH', type=int, default=5, help='Min Length of sequence')
parser.add_argument('--MAX_LENGTH', type=int, default=200, help='Max Length of sequence')
parser.add_argument('--lang1', type=str, default="en", help='Input Language')
parser.add_argument('--lang2', type=str, default="fr", help='Target Language')
parser.add_argument('--use_cuda', action='store_false', help='IF USE CUDA')
parser.add_argument('--teacher_forcing_ratio', type=int, default=0.5, help='Teacher forcing ratio for encoder')
parser.add_argument('--hidden_size', type=int, default=256, help='Size of hidden layer')
parser.add_argument('--n_iters', type=int, default=3000, help='Number of single iterations through the data')
parser.add_argument('--learning_rate', type=int, default=0.01, help='Learning rate (for both, encoder and decoder)')
parser.add_argument('--n_layers', type=int, default=1, help='Number of layers (for both, encoder and decoder)')
parser.add_argument('--dropout_dec_p', type=int, default=0.1, help='Dropout (%) in the decoder')
parser.add_argument('--model_type', type=str, default="seq2seq", help='Model type (and ending of files)')
parser.add_argument('--main_data_dir', type=str, default= "/Users/eduardofierro/Google Drive/TercerSemetre/NLP/ProjectOwn/Data/Model_ready/", help='Directory where data is saved (in folders tain/dev/test)')
parser.add_argument('--out_dir', type=str, default="", help="Directory to save the models state dict")
opt = parser.parse_args()
print(opt)

######## Comet ML ########

experiment = Experiment(api_key="00Z9vIf4wOLZ0yrqzdwHqttv4", log_code=True)
hyper_params = vars(opt)
experiment.log_multiple_params(hyper_params)

#################################
# Languages classes and imports #
#################################

PAD_token = 0
SOS_token = 1
EOS_token = 2


class Lang:
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

def read_langs(lang1, lang2, term="txt", reverse=False):
    print("Reading lines...")

    # Read the file and split into lines

    # Attach the path here for the source and target language dataset
    filename = '%s/train/%s-%s.%s' % (opt.main_data_dir, lang1, lang2, term)
    # This creats the file directory name whichis used below

    # lines contains the data in form of a list 
    lines = open(filename).read().strip().split('\n')

    # Split every line into pairs
    pairs = [[s for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length

def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def prepare_data(lang1_name, lang2_name, do_filter=True, MIN_LENGTH=opt.MIN_LENGTH, MAX_LENGTH=opt.MAX_LENGTH, reverse=False):

    # Get the source and target language class objects and the pairs (x_t, y_t)
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, term=opt.model_type, reverse=reverse)
    print("Read %d sentence pairs" % len(pairs))
 
    if do_filter==True:
        pairs = filterPairs(pairs, MAX_LENGTH)
        print("Filtered to %d pairs" % len(pairs))
    else: 
        print("Pairs not filtered...")
    
    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])
    
    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    return input_lang, output_lang, pairs
    
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if opt.use_cuda: 
        var = var.cuda()
    return var

def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)
    
################################
# Main model encoder - decoder #
################################  
    
# Both classes form Lab - Week 9 (Lab8)   

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if opt.use_cuda:
            return result.cuda()
        else:
            return result
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=opt.MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if opt.use_cuda:
            return result.cuda()
        else:
            return result    
            
############################
# Training & training loop #
############################
            
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    
def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=opt.MAX_LENGTH):
    
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if opt.use_cuda else encoder_outputs
   
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if opt.use_cuda else decoder_input
    
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < opt.teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if opt.use_cuda else decoder_input
            
            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length
    
def trainIters(encoder, decoder, n_iters, pairs, learning_rate=0.01, 
               print_every=5000, save_every=5000):
    
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    
    # Optimizers = ADAM in Chung, Cho and Bengio 2016
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [variables_from_pair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]
 
        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            experiment.log_metric("Train loss", print_loss_avg)

        if iter % save_every == 0: 
            torch.save(encoder.state_dict(), "{}/saved_encoder_{}.pth".format(opt.out_dir, iter))
            torch.save(decoder.state_dict(), "{}/saved_decoder_{}.pth".format(opt.out_dir, iter))
        
#########
# Train #
#########

input_lang, output_lang, pairs = prepare_data(opt.lang1, opt.lang2)

encoder1 = EncoderRNN(input_lang.n_words, opt.hidden_size)
attn_decoder1 = AttnDecoderRNN(opt.hidden_size, output_lang.n_words,
                               opt.n_layers, dropout_p=opt.dropout_dec_p)

if opt.use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

trainIters(encoder1, attn_decoder1, n_iters=opt.n_iters, pairs=pairs, learning_rate=opt.learning_rate, print_every=100)

torch.save(encoder1.state_dict(), "{}/saved_encoder_final.pth".format(opt.out_dir))
torch.save(attn_decoder1.state_dict(), "{}/saved_decoder_final.pth".format(opt.out_dir))         
