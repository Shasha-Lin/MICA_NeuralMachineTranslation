from main_model_2 import *

print("success")

#########
# Test  #
#########

# python test_main_model_2.py --main_data_dir ../../Model2_ready --model_type bpe2char --learning_rate 0.001 --enc_checkpoint checkpoints/saved_encoder_3000.pth --dec_checkpoint checkpoints/saved_decoder_3000.pth --outlang_checkpoint checkpoints/output_lang.pth

def test_model(input_variable, target_variable, encoder, decoder, criterion, out_lang, max_length=opt.MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder.eval()
    decoder.eval()
    
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size), volatile=True)
    encoder_outputs = encoder_outputs.cuda() if opt.use_cuda else encoder_outputs
   
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]), volatile=True)
    decoder_input = decoder_input.cuda() if opt.use_cuda else decoder_input
    
    decoder_hidden = encoder_hidden
    
    total_decoder_output = []
    # Without teacher forcing: use its own predictions as the next input
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        topv, topi = decoder_output.data.topk(2)
        ni = topi[0][0]
        
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if opt.use_cuda else decoder_input

        # loss += criterion(decoder_output, target_variable[di])
        # Choose top word from output
        if ni == 2:
            print("appending EOS")
            total_decoder_output.append('<EOS>')
            break
        else:
            total_decoder_output.append(out_lang.vocab[ni-3][0])
    debug = 0
    return total_decoder_output

def translate():
    out_lang_saved = torch.load(opt.outlang_checkpoint)
    input_lang, output_lang, pairs = prepare_data(opt.lang1, opt.lang2, char_output=target_char)
    encoder1 = EncoderRNN(input_lang.n_words, opt.hidden_size)
    attn_decoder1 = AttnDecoderRNN(opt.hidden_size, output_lang.n_words,
                                   opt.n_layers, dropout_p=opt.dropout_dec_p)
    if opt.use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()
    
    encoder1.load_state_dict(torch.load(opt.enc_checkpoint))
    attn_decoder1.load_state_dict(torch.load(opt.dec_checkpoint))
    n_iters = 5
    
    training_pairs = [variables_from_pair(random.choice(pairs), input_lang, output_lang)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()
    
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]
 
        dec_results = test_model(input_variable, target_variable, encoder1,
                     attn_decoder1, criterion, output_lang)
        print(len(dec_results))
        print(dec_results)
        #print(output_lang.detokenize(dec_results))
    print("success 2")

    #trainIters(encoder1, attn_decoder1, n_iters=opt.n_iters, pairs=pairs, learning_rate=opt.learning_rate, print_every=opt.print_every, save_every=opt.save_every)

    #torch.save(encoder1.state_dict(), "{}/saved_encoder_final.pth".format(opt.out_dir))
    #torch.save(attn_decoder1.state_dict(), "{}/saved_decoder_final.pth".format(opt.out_dir))         

if __name__ == '__main__':
    translate()