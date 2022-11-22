import torch
import torch.nn as nn
from lstm_mdl import myLstm
class RNN(nn.Module):
    # RNN model is composed of three parts: a word embedding layer, a rnn network and a output layer
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding
    # The network has input of each word embedding and output a hidden feature corresponding to each word embedding
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary
    # feel free to change the init arguments if necessary
    def __init__(self, nvoc, ninput, nhid, nlayers):
        super(RNN, self).__init__()
        self.rnn_type = 'GRU'
        self.drop = nn.Dropout(0.5)

        # self.embed change input size BxL into BxLxE
        self.embed = nn.Embedding(nvoc, ninput) 
        
        # WRITE CODE HERE witnin two '#' bar                                              #
        # Construct you RNN model here. You can add additional parameters to the function #
        ###################################################################################
        #self.rnn = nn.GRU(input_size=ninput, hidden_size=nhid, num_layers=nlayers, batch_first=False)
        self.rnn = myLstm(input_sz=ninput, hidden_sz=nhid)

        ###################################################################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        init_uniform = 0.1
        self.embed.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
    
    # feel free to change the forward arguments if necessary
    def forward(self, input):
        embeddings = self.embed(input)
        embeddings = self.drop(embeddings)

        # WRITE CODE HERE within two '#' bar                                             #
        # With embeddings, you can get your output here.                                 #
        # Output has the dimension of sequence_length * batch_size * number of classes   #
        ##################################################################################
        output, hidden = self.rnn(embeddings)

        ##################################################################################

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        out_f = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded, hidden
    def init_hidden(self, batchsize, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.nlayers,batchsize, self.nhid), requires_grad=requires_grad),
                    weight.new_zeros((self.nlayers,batchsize, self.nhid), requires_grad=requires_grad))
        else:
            return weight.new_zeros((self.nlayers, batchsize, self.nhid), requires_grad=requires_grad)



# WRITE CODE HERE within two '#' bar                                                      #
# your transformer for language modeling implmentation here                               #
###########################################################################################
from Utils import PositionalEncoding
from torch.nn import TransformerEncoderLayer ,TransformerEncoder
import math
class LMTransformer(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid,
                 nlayers, dropout = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz) :
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

###########################################################################################
