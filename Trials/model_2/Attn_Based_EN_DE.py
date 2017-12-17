# Code source : https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

from Encoder_Decoder import *
import torch

# The Attention based decoder is also structured based on this paaper: https://arxiv.org/pdf/1409.0473.pdf

# If GPU being used, set TRUE else FALSE:
USE_CUDA = torch.cuda.is_available()

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size_enc, hidden_size_dec, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = hidden_size_dec
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size_dec)
        self.attn = nn.Linear(self.hidden_size_dec * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size_enc * 2, self.hidden_size_dec)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size_dec, self.hidden_size_dec)
        self.out = nn.Linear(self.hidden_size_dec, self.output_size)

    def forward(self, input_data, hidden, encoder_outputs):
        embedded = self.embedding(input_data).view(1, 1, -1)
        embedded = self.dropout(embedded)
        print("calc attn_weights")
        print(embedded[0].size())
        print(hidden[0].size())
        print(torch.cat((embedded[0], 
                                 hidden[0])).size())
        print(self.attn())
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], 
                                 hidden[0].view(1,-1,1)), 1)))
        print("calc attn_applied")
        print(attn_weights.unsqueeze(0).size())
        print(encoder_outputs.unsqueeze(0).size())
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
        if use_cuda:
            return result.cuda()
        else:
            return result
