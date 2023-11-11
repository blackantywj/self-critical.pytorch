from .captioning.utils.utils import *
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        self.dropout = nn.Dropout(p=0.1)
        self.layernom = LayerNorm(self.d_model)
        pe = torch.zeros(5000, d_model)
        position = torch.arange(0, 5000).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        a = self.lut(x) * math.sqrt(self.d_model)
        a = self.layernom(a + self.pe[:, :a.size(1)] )
        a = self.dropout(a)
        return a

class bit_Embeddings(nn.Module):
    def __init__(self, d_model, vocab,bit_dim ):
        super(bit_Embeddings, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=0.1)
        self.layernom = LayerNorm(self.d_model)
        self.vocab_size = vocab
        self.bit_dim = bit_dim
        self.bit_scale = 1.0

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        x = x.view(batch_size, seq_length, 1, 1)
        input_ids_bit = decimal_to_bits(x, vocab_size=self.vocab_size, bits=self.bit_dim) * self.bit_scale
        return input_ids_bit
    
class diff_Embeddings(nn.Module):
    def __init__(self, d_model):
        super(diff_Embeddings, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=0.1)
        self.layernom = LayerNorm(self.d_model)
        # self.L1 = nn.Linear(14,512,bias=False)
        self.max_len = 50
        self.position_embeddings = nn.Embedding(50, d_model)
        self.time_embeddings =nn.Sequential(
    LearnedSinusoidalPosEmb(),
    nn.Linear(in_features=257, out_features=512, bias=True),
    nn.SiLU(),
    nn.Linear(in_features=512, out_features=512, bias=True)
)
    def forward(self, x,time,position_ids=None):
        x_size = x.size(1)
        if True:
            position_ids = torch.arange(x_size, dtype=torch.long, device=x.device).view(1, -1)
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        time_e = self.time_embeddings(time).unsqueeze(1)
        x = x + time_e
        x = self.dropout(self.layernom(x))
        return x