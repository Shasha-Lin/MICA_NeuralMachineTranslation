# defines training loop 
# Code source : https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
import time
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

# from Attn_Based_EN_DE import *
from masked_cross_entropy import masked_cross_entropy
from data_for_modeling import random_batch, EOS_token, SOS_token

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


def train(use_cuda, input_variable, input_lengths, target_variable, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, max_length, batch_size, teacher_forcing_ratio=.5):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, input_lengths, None)
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs[t] = decoder_output
            decoder_input = target_variable[t]  # Next input is current target
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            if ni == EOS_token:
                break
                # Loss calculation and backpropagation
    loss = masked_cross_entropy(all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths)
    loss.backward()
    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], ec, dc


def edu_trainIters(use_cuda, input_lang, output_lang, encoder, decoder, n_iters, pairs, pairs_eval, loss_criterion,
               learning_rate=0.01, 
               print_every=5000, save_every=5000, eval_every=10000, opt=None, char=False):
    
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    
    # Optimizers = ADAM in Chung, Cho and Bengio 2016
    if opt.optimizer == "Adam":
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    elif opt.optimizer == "SGD":
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)  
    else: 
        raise ValueError('Optimizer options not found: Select SGD or Adam') 
    
    #training_pairs = [variables_from_pair(random.choice(pairs))
    #                  for i in range(n_iters)]
    training_pairs = [variables_from_pair(random.choice(pairs), input_lang, output_lang, char=char, use_cuda=use_cuda)
                      for i in range(n_iters)]
                      
    criterion = loss_criterion

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
            
        if iter % eval_every == 0: 
            prediction = evaluate_dev(input_lang, output_lang, encoder, decoder, pairs_eval)
            target_eval = [x[1] for x in pairs_eval]
            bleu_corpus = bleu_score.corpus_bleu(target_eval, prediction)
            experiment.log_metric("BLEU score", bleu_corpus)
            print("NLTK's Blue score: {} at iter {}".format(round(bleu_corpus, 2), iter))
            evaluateRandomly(input_lang, output_lang, encoder1, attn_decoder1, max_length=opt.MAX_LENGTH_TARGET, n=5)
            




def trainIters(use_cuda, encoder, decoder, n_iters, pairs, in_lang, out_lang, pairs_eval, opt=None, outdir='.', learning_rate=0.01, print_every=100, save_every=5000, eval_every=1000, char=False):
    # defining some variables from opt object
    max_length =opt.MAX_LENGTH_TARGET
    teacher_forcing_ratio = opt.teacher_forcing_ratio
    batch_size = opt.batch_size
    
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    # Optimizers = ADAM in Chung, Cho and Bengio 2016
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # training_pairs = [variables_from_pair(random.choice(pairs), in_lang, out_lang, char=char, use_cuda=use_cuda) for i in range(n_iters)]
    # training_pairs = [random.choice(pairs) for i in range(n_iters)]
    training_pairs = pairs
    encoder.train()
    decoder.train()
    for iter in range(1, n_iters + 1):
        input_batches, input_lengths, target_batches, target_lengths = \
            random_batch(use_cuda, batch_size, training_pairs, in_lang, out_lang, char_output=char)
        loss = train(use_cuda, input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, max_length, batch_size, teacher_forcing_ratio)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            # experiment.log_metric("Train loss", print_loss_avg)

        if iter % save_every == 0:
            torch.save(encoder.state_dict(), "{}/saved_encoder_{}.pth".format(out_dir, iter))
            torch.save(decoder.state_dict(), "{}/saved_decoder_{}.pth".format(out_dir, iter))

        if iter % eval_every == 0:
            encoder.train(False)
            decoder.train(True)
            prediction = evaluate_dev(input_lang, output_lang, encoder, decoder, pairs_eval)
            target_eval = [x[1] for x in pairs_eval]
            bleu_corpus = bleu_score.corpus_bleu(target_eval, prediction)
            experiment.log_metric("BLEU score", bleu_corpus)
            evaluateRandomly(input_lang, output_lang, encoder1, attn_decoder1, max_length=max_length, n=5)
            encoder.train()
            decoder.train()

def indexes_from_sentence(lang, sentence, char=False):
    if char:
        return [lang.word2index(word) for word in sentence.split(' ')]
    else:
        return [lang.word2index[word] for word in sentence.split(' ')]

    
def variable_from_sentence(lang, sentence, use_cuda=False, char=False):
    indexes = indexes_from_sentence(lang, sentence, char)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda: 
        var = var.cuda()
    return var

            
def variables_from_pair(pair, input_lang, output_lang, char=False, use_cuda=False):
    input_variable = variable_from_sentence(input_lang, pair[0], use_cuda=use_cuda)
    target_variable = variable_from_sentence(output_lang, pair[1], char=char, use_cuda=use_cuda)
    return (input_variable, target_variable)

