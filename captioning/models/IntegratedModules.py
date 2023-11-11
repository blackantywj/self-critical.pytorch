class posEncoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(posEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.norm1 = LayerNorm(layer.size)
        self.dropout = nn.Dropout(0.1)
        # self.sublayer1 = clones(SublayerConnection(1024, dropout=0.1), 1)
        self.latent = Latent(layer.size)
        self.houyan1 = MultiHeadedAttention(8, 512, dropout=0.1)
        self.houyan2 = MultiHeadedAttention(8, 512, dropout=0.1)
        self.feed_forward = PositionwiseFeedForward_11(layer.size * 2, 2048, dropout=0.1)
        self.feed_forward1 = PositionwiseFeedForward_11(layer.size * 2, 2048, dropout=0.1)
        # self.feed = PositionwiseFeedForward(512, 2048, dropout=0.1)
        self.av1 = AverageSelfAttention(layer.size)
        self.av2 = AverageSelfAttention(layer.size)
        # self.linear = Generator(512*2,7488)
    ''''
    pos_h, memory (VN), src_mask, pos_mask, target_mask, train
    '''
    def forward(self, x,  memory, src_mask, seq_mask,tgt_mask, train=True):
        if train == True:
            x_norm1 = self.houyan1(x, x, x, seq_mask)
            x_norm = self.dropout(x_norm1+x)
            x_norm1 = self.houyan2(x_norm, memory, memory,src_mask)
            x_norm = self.dropout(x_norm1 + x_norm)
            x_norm = self.av1(x_norm, seq_mask.squeeze(1))
            x_norm_p = self.av2(memory, src_mask.squeeze(1))
            kl_loss, z = self.latent(x_norm_p, x_norm, train)
        if train == False and x is not None :
            x_norm1 = self.houyan1(x, x, x, seq_mask)
            x_norm = self.dropout(x_norm1+x)
            x_norm1 = self.houyan2(x_norm,memory,memory,src_mask)
            x_norm = self.dropout(x_norm1 + x_norm)
            x_norm = self.av1(x_norm,seq_mask.squeeze(1)) # houyan
            x_norm_p = self.av2(memory,src_mask.squeeze(1)) # xianyan
            kl_loss, z = self.latent(x_norm_p, x_norm, True)
            return self.norm(z), kl_loss, z
        if x is None:
            x_norm_p = self.av2(memory, src_mask.squeeze(1))
            x_norm = None
            kl_loss, z = self.latent(x_norm_p, x_norm, train=False)
            return self.norm(z), kl_loss, z
        return self.norm(z), kl_loss, z

'''
变分隐空间，实际包括编码器
'''
class Latent(nn.Module):
    def __init__(self,dim):
        super(Latent, self).__init__()
        self.mean = PositionwiseFeedForward(dim, 2048, dropout=0)
        self.mean_1 = PositionwiseFeedForward(dim, 2048, dropout=0)
        self.var = PositionwiseFeedForward(dim, 2048, dropout=0)
        self.var_1 = PositionwiseFeedForward(dim, 2048, dropout=0)
        self.mean_p = PositionwiseFeedForward_11(dim*2, 2048, dropout=0.1)
        self.var_p = PositionwiseFeedForward_11(dim*2, 2048, dropout=0.1)
    def forward(self, x, x_p, train=True):
        mean = self.mean(x)
        log_var = self.var(x)
        eps = torch.randn(x.size())
        std = torch.exp(0.5 * log_var)
        eps = eps.cuda()
        z = eps * std + mean
        kld_loss = 0
        if x_p is not None:
            x_p = x_p.squeeze(1)
            mean_p = self.mean_1(x_p)
            log_var_p = self.var_1(x_p)
            kld_loss = gaussian_kld(mean_p, log_var_p, mean, log_var)
            kld_loss = torch.mean(kld_loss)
        if train:
            std = torch.exp(0.5 * log_var_p)
            eps = eps.cuda()
            z = eps * std + mean_p
            # z =  mean_p
        return kld_loss, z  

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask=None):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x,tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m,src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0



