# Code source : https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

import torch
import io
from data_loader import *
import matplotlib.pyplot as plt
import visdom
vis = visdom.Visdom()
# Evaluation is mostly the same as training, but there are no targets. Instead we always feed the decoder's predictions back to itself. 
# Every time it predicts a word, we add it to the output string. If it predicts the EOS token we stop there. We also store the decoder's attention outputs for each step to display later.

def indexes_from_sentence(lang, sentence, char=False):
    if char:
        return [lang.word2index(word) for word in sentence.split(' ')]
    else:
        return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(use_cuda, lang, sentence, char=False):
    input_lengths = [len(input_seq)]
    indexes = indexes_from_sentence(lang, sentence, char)
    indexes.append(EOS_token)
    indexes = [indexes]
    input_batches = Variable(torch.LongTensor(indexes), volatile=True).transpose(0, 1)
    if use_cuda:
        input_batches = input_batches.cuda()
    return input_batches, input_lengths

def evaluate(use_cuda, in_lang, out_lang, encoder, decoder,
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