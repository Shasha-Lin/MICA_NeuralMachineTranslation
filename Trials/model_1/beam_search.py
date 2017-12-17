# BEAM SEARCH:
    # A dictinary is maintained which keeps track of the sequence within top first k probabilities
    # If k = 1, it is same as greedy search. 
    # The dictinary is of the form {sequence: [probability, decoder_hidden, decoder_attention] }
    # The dictinary data type is  {string: [int, torch.Size([num_directions, batch_size, hidden_size]), torch.Size([max_length+1, max_length+1])]}


def update_dictionary(target_sequence, topv, topi, key, dec_hidden, decoder_attns):
    if len(target_sequence) == 0:
        for i in range(len(topi)):
            target_sequence.update({str(topi[i]) : [topv[i], dec_hidden, decoder_attns] })
    else:
        prev_val = target_sequence[key][0]
        for i in range(len(topi)):
            target_sequence.update({key+"-"+str(topi[i]) : [topv[i]*prev_val, dec_hidden, decoder_attns] })
        del[target_sequence[key]]
        
def get_seq_through_beam_search(max_length, decoder, decoder_input, decoder_hidden, decoder_attentions, encoder_outputs, kmax ):
    target_sequence = dict()
    
    # Run through decoder
    for di in range(max_length):
        
        if di == 0:
            decoder_output, decoder_hidden, decoder_attention = decoder( decoder_input, decoder_hidden, encoder_outputs )
            topv, topi = decoder_output.data.topk(kmax)
            topv = topv[0].numpy()
            topi = topi[0].numpy()
            decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data
            update_dictionary(target_sequence, topv, topi, None, decoder_hidden, decoder_attentions)
        else:
            temp = target_sequence.copy()
            keys = list(temp.keys())
            for i in range(len(keys)):
                inp = int(keys[i].split("-")[-1] if len(keys[i]) > 1 else keys[i])
                if inp != EOS_token:
                    dec_input = Variable(torch.LongTensor([inp]))
                    decoder_output, dec_hidden, decoder_attention = decoder( dec_input, temp[keys[i]][1], encoder_outputs )
                    topv, topi = decoder_output.data.topk(kmax)
                    topv = topv[0].numpy()
                    topi = topi[0].numpy()
                    dec_attns = temp[keys[i]][2]
                    dec_attns[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data
                    update_dictionary(target_sequence, topv, topi, keys[i], dec_hidden, dec_attns)
        
        # Sort the target_Sequence dictionary to keep top k sequences only
        target_sequence = dict(sorted(target_sequence.items(), key=lambda x: x[1][0], reverse=True)[:kmax])
     
    # Get the sequence, decoder_attentions with maximum probability
    pair = sorted(target_sequence.items(), key=lambda x: x[1][0], reverse=True)[:1][0]
    seq = pair[0]
    decoder_attentions = pair[1][2]
    
    # Get the decoded words:
    decoded_words_indices = seq.split("-")
    decoded_words = [output_lang.index2word[int(i)] for i in decoded_words_indices]
    if int(decoded_words_indices[-1]) != EOS_token:
        decoded_words.append('<EOS>')
    
    return decoded_words, decoder_attentions