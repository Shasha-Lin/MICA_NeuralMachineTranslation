# Code source : https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

from Encoder import *
import torch

# The Attention based decoder is also structured based on this paaper: https://arxiv.org/pdf/1409.0473.pdf

# If GPU being used, set TRUE else FALSE:
USE_CUDA = torch.cuda.is_available()
MAX_LENGTH = 20
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size_enc, hidden_size_dec, output_size, n_layers=1, dropout=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = hidden_size_dec
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout
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
        if USE_CUDA:
            return result.cuda()
        else:
            return result
