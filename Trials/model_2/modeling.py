# File used to train a model over multiple epochs
# Code source : https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
# no file named evaluation
import torch
# from evaluation_and_attention_visualization import *
# adding this import for clarity 
import io
import torchvision
from PIL import Image
import visdom
vis = visdom.Visdom()
import torch.nn as nn
from tokenizer import CharTokenizer
# If GPU being used, set TRUE else FALSE:
USE_CUDA = torch.cuda.is_available()

MAX_LENGTH = 100
from training import *
from data_for_modeling import *
from Attn_Based_EN_DE import *

def main():

    ##########################################################################
    ###### PART-I : Data Formation using scripts in data_for_modeling.py #####
    ##########################################################################

    input_lang, output_lang, pairs = prepare_data('en', 
                                                  'fr', 
                                                  False, 
                                                  path="/Users/millie/Documents/NLP_fall_2017/project/Model2_ready/train",
                                                  term='bpe2char',
                                                  char_output=True
                                                 )

    # TRIMMING DATA:
    # Trimming is optional but could be done to reduce the data size and make processing faster
    # Removes words with frequency < 5

    MIN_COUNT = 5

    input_lang.trim(MIN_COUNT)
    output_lang.trim(MIN_COUNT)


    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1].lower()
        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in input_lang.word2index:
                keep_input = False
                break

        for word in list(output_sentence):
            if word not in dict(output_lang.vocab) and word != " ":
                keep_output = False
                break

        # Remove if pair doesn't match input and output conditions
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    pairs = keep_pairs



    ##########################################################################
    ######     PART-II : Setup Configuration for training the data       #####
    ##########################################################################



    # Configure models
    # attn_model = 'dot'
    hidden_size_enc = 512
    hidden_size_dec = 1024
    n_layers = 2
    dropout = 0.1
    batch_size = 128
    # batch_size = 50

    # Configure training/optimization
    # clip = 50.0
    clip = 1.0 # Based on our paper, clipping gradient norm is 1
    teacher_forcing_ratio = 0 #0.5
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_epochs = 50000
    epoch = 0
    plot_every = 20
    print_every = 100
    evaluate_every = 10000 # We check the validation in every 10,000 minibatches
        # Initialize models
    encoder = EncoderRNN(input_lang.n_words, hidden_size_enc, n_layers, dropout=dropout)
    # decoder = BahdanauAttnDecoderRNN(hidden_size_dec, output_lang.n_words, n_layers, dropout_p=dropout)
    decoder = AttnDecoderRNN( hidden_size_enc=hidden_size_enc,
                             hidden_size_dec=hidden_size_dec, 
                             output_size=output_lang.n_words, 
                             n_layers=n_layers, 
                             dropout_p=dropout)
    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    criterion = nn.NLLLoss()

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    # import sconce
    # job = sconce.Job('seq2seq-translate', {
    #     'attn_model': attn_model,
    #     'n_layers': n_layers,
    #     'dropout': dropout,
    #     'hidden_size_enc': hidden_size_enc,
    #     'hidden_size_dec': hidden_size_dec,
    #     'learning_rate': learning_rate,
    #     'clip': clip,
    #     'teacher_forcing_ratio': teacher_forcing_ratio,
    #     'decoder_learning_ratio': decoder_learning_ratio,
    # })
    # job.plot_every = plot_every
    # job.log_every = print_every

    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every



    ##########################################################################
    ######                         PART-III : Modeling                   #####
    ##########################################################################

    ecs = []
    dcs = []
    eca = 0
    dca = 0

    while epoch < n_epochs:
        epoch += 1

        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size, 
                                                                                    pairs,
                                                                                    input_lang,
                                                                                    output_lang,
                                                                                    char_output=True
                                                                                   )

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer,
            criterion,
            batch_size=batch_size
        )

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        eca += ec
        dca += dc

    #     job.record(epoch, loss)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % evaluate_every == 0:
            evaluate_randomly()

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            # TODO: Running average helper
            ecs.append(eca / plot_every)
            dcs.append(dca / plot_every)
            ecs_win = 'encoder grad (%s)' % hostname
            dcs_win = 'decoder grad (%s)' % hostname
            vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win})
            vis.line(np.array(dcs), win=dcs_win, opts={'title': dcs_win})
            eca = 0
            dca = 0
        # Initialize models
        encoder = EncoderRNN(input_lang.n_words, hidden_size_enc, n_layers, dropout=dropout)
        decoder = BahdanauAttnDecoderRNN(hidden_size_dec, output_lang.n_words, n_layers, dropout_p=dropout)

        # Initialize optimizers and criterion
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
        criterion = nn.NLLLoss()

        # Move models to GPU
        if USE_CUDA:
            encoder.cuda()
            decoder.cuda()

        import sconce
        job = sconce.Job('seq2seq-translate', {
            'attn_model': attn_model,
            'n_layers': n_layers,
            'dropout': dropout,
            'hidden_size_enc': hidden_size_enc,
            'hidden_size_dec': hidden_size_dec,
            'learning_rate': learning_rate,
            'clip': clip,
            'teacher_forcing_ratio': teacher_forcing_ratio,
            'decoder_learning_ratio': decoder_learning_ratio,
        })
        job.plot_every = plot_every
        job.log_every = print_every

        # Keep track of time elapsed and running averages
        start = time.time()
        plot_losses = []
        print_loss_total = 0 # Reset every print_every
        plot_loss_total = 0 # Reset every plot_every



        ##########################################################################
        ######                         PART-III : Modeling                   #####
        ##########################################################################

        ecs = []
        dcs = []
        eca = 0
        dca = 0

if __name__ == "__main__" :
    main()