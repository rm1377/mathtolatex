import torch 
nn = torch.nn

MAX_LEN = 100
IMG_HSIZE = 240
IMG_WSIZE = 320 
HiddenSize = 512
EmbedSize = 32
AttnSize = 64

class CNNModel(nn.Module):
    def __init__(self, last_channel):
        super(CNNModel, self).__init__() 
        self.base_model_ = nn.Sequential(
            nn.Conv2d(1, 64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d([1,2],[1,2]),

            nn.Conv2d(256, last_channel,3,1,1),
            nn.BatchNorm2d(last_channel),
            nn.ReLU(True),
            nn.MaxPool2d([2,1],[2,1]),

            nn.Conv2d(last_channel, last_channel,3,1,1),
            nn.BatchNorm2d(last_channel),
            nn.ReLU(True),
            
        )

    def forward(self, x):
        x = self.base_model_(x) 
        x = x.permute(0,2,3,1)
        return x





class Attention(nn.Module):
    def __init__(self, hidden_size,attn_size):
        super(Attention, self).__init__()
        self.enc_attn = nn.Linear(hidden_size,attn_size)
        self.dec_attn = nn.Linear(hidden_size, attn_size) 
        self.relu = nn.ReLU()
        self.attn = nn.Linear(attn_size, 1)
        self.soft = nn.Softmax(1)
    def forward(self, encoder_output, decoder_hidden):
        enc_attn = self.enc_attn(encoder_output)
        dec_attn = self.dec_attn(decoder_hidden).unsqueeze(1)
        f1 =  self.relu(enc_attn+dec_attn)
        attn_weights = self.attn(f1)
        attn_weights = self.soft(attn_weights)
        context = (attn_weights*encoder_output).sum(1)
        return context, attn_weights



class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,embed_size=128, attn_size=64):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embed_size)
        dec_input_size = embed_size + hidden_size
        self.decoder_step = nn.LSTMCell(dec_input_size, hidden_size) #input of shape (batch, input_size), h_0 of shape (batch, hidden_size),c_0 of shape (batch, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention(hidden_size, attn_size)

    def forward(self, input, encoder_output,state):
        # h_0, c_0 = state
        embed_vector = self.embedding(input)
        context_vector, attn_weights = self.attn(encoder_output, state[0])
        decoder_input = torch.cat([context_vector, embed_vector],1)
        state = self.decoder_step(decoder_input, state)
        output = self.out(state[0])
        return output, state

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)

