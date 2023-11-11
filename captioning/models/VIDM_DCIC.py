# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .captioning.utils.utils import *
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
    
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, posencoder, decoder, decoder2,  src_embed, bit_embed1, bit_embed2, diff_embed, generator, gs_diff
                 ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.posencoder = posencoder
        self.avaattention = AverageSelfAttention(512)
        self.decoder = decoder
        self.decoder2 = decoder2
        self.src_embed = src_embed
        self.posembed = bit_embed1
        self.capembed = bit_embed2
        self.norm_e1 = LayerNorm(512)
        self.norm_e2 = LayerNorm(512)
        self.diff_embed = diff_embed
        self.generator = nn.Linear(512, 9488, bias=True)
        self.generator_pos = generator
        self.norm1 = LayerNorm(512)
        self.dropout = nn.Dropout(0.1)
        self.li1 = nn.Linear(4, 512,bias=False)
        self.li2 = nn.Linear(14, 512, bias=False)
        # self.li3 = nn.Linear(20, 512, bias=Fa)
        self.gs_diff = gs_diff
        self.vac = 9488
        self.norm_p1 = LayerNorm(512)
        self.norm_p = LayerNorm(512)
        self.pos_att = MultiHeadedAttention(8, 512, dropout=0.1)
        # with torch.no_grad():
        #     self.generator.weight = self.embed.weight
    def gene(self,hidden_repr):
        # preds = self.generator(hidden_repr)
        text_emb = hidden_repr
        emb_norm = (self.generator.weight ** 2).sum(-1).view(-1, 1)  # vocab
        text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
        arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
        dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(self.generator.weight,
                                                                    text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
        scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                                 hidden_repr.size(1))  # vocab, bsz*seqlen
        scores = -scores.permute(1, 2, 0).contiguous()

        return scores
    def generate (self, out):
        outputs1 = self.generator(out)
        buffer_probs = nn.Softmax(dim=-1)(outputs1)
        vocab_inds = torch.arange(0, self.vac).long().view(1, self.vac, 1, 1).cuda()
        vocab_bit_buffer = decimal_to_bits(vocab_inds, vocab_size=self.vac, bits=14)
        logits_w = torch.matmul(buffer_probs, vocab_bit_buffer.expand(buffer_probs.size(0), -1, -1))
        return outputs1,logits_w
    
    def generate1 (self, out):
        outputs1 = self.generator(out)
        buffer_probs = nn.Softmax(dim=-1)(outputs1)
        vocab_inds = torch.arange(0, 9488).long().view(1, 9488, 1, 1).cuda()
        vocab_bit_buffer = decimal_to_bits(vocab_inds, vocab_size=9488, bits=14)
        logits_w = torch.matmul(buffer_probs, vocab_bit_buffer.expand(buffer_probs.size(0), -1, -1))
        return outputs1,logits_w
    def loss_bit(self,target,pre,mask):

        logits = pre[mask, :]
        targets = target[mask, :]
        loss = F.mse_loss(logits, targets)
        return loss
    def sample(self,image,bit_noise,time,pos,img_masks,text_masks=None):
        bit_noise =self.li1(bit_noise)
        latent = self.diff_embed(bit_noise,time,pos)
        out1 = self.decode(image, img_masks, latent, text_masks)
        out_pro ,bit_pre= self.generate(out1)
        return bit_pre
    # u_token_id 是语句中的token序列编码
    def sample1(self, image, bit_noise, time, pos, img_masks, text_masks=None):
        
        # pos = self.posembed(pos)
        # pos_1 = self.pos_att(pos,pos,pos)
        # pos = self.norm_p(pos + pos_1)
        bit_noise =self.li2(bit_noise)
        latent = self.diff_embed(bit_noise,time)
        bit_noise = latent + pos
        latent = self.norm_p1(bit_noise)
        out1 = self.decode(image, img_masks, latent, text_masks)
        out_pro ,bit_pre= self.generate(out1)
        return bit_pre
    
    # inputs 里面到底是什么
    def sample_copy(self, image, bit_noise, time, pos, img_masks, text_masks=None):
        
        # pos = self.posembed(pos)
        # pos_1 = self.pos_att(pos,pos,pos)
        # pos = self.norm_p(pos + pos_1)
        bit_noise =self.li2(bit_noise)
        latent = self.diff_embed(bit_noise,time)
        bit_noise = latent + pos
        latent = self.norm_p1(bit_noise)
        out1 = self.decode(image, img_masks, latent, text_masks)
        out_pro ,bit_pre= self.generate(out1)
        return bit_pre    
    
    def _diff_decoder_forward(self, inputs):
        te_out = self.token_embed(inputs)
        
    def posencode(self, pos, memory, src_mask, target_mask, train=True):
        #and not (pos[0].equal(pos[12]))
        if pos is not None:
            pos_h = self.posembed(pos)
            pos_mask  = (pos != 0).unsqueeze(1)
            out = self.posencoder(pos_h, memory, src_mask, pos_mask, target_mask, train)
        if pos is None:
            pos_mask = torch.ones(memory.size(0),1,32).cuda()
            pos_h = None
            out = self.posencoder(pos_h, memory, src_mask, pos_mask,target_mask,False)
        return out
    def forward(self, src, pos,  tgt, src_mask, tgt_mask, stage):
        "Take in and process masked src and target sequences."
        # combine
        if stage == 0:
            # image encoder src -> VN
            image = self.encode(src, src_mask)
            
            # VN + MHA(embedding(pos)) do cross attention -> norm(z)/kl loss/z 
            pos_kl = self.posencode(pos, image, src_mask, tgt_mask)
            # z as pos
            pos = pos_kl[-1].unsqueeze(1)
            
            # caption decoder
            x_start = self.capembed(tgt)
            x_t,t = noise_sample(x_start)
            x_t = self.li2(x_t)
            latent = self.diff_embed(x_t,t)
            latent = latent + pos.expand_as(latent)
            x_t = self.norm_p1(latent)
            out1 = self.decode(image, src_mask, x_t)
            out_pro ,bit_pre= self.generate1(out1)
            
            # masked pos decoder
            out_pos = self.decoder2(pos.expand_as(latent), image.detach(), src_mask, tgt_mask)
            out_pos = self.generator_pos(out_pos)
            
            # compute bit loss
            tgt_mask = tgt_mask.squeeze(1)
            bitloss = self.loss_bit(x_start, bit_pre, tgt_mask.squeeze(1))
            return bitloss, out_pro, tgt_mask, out_pos, pos_kl[1]
        if stage == 1:
            image = self.encode(src, src_mask)
            pos_kl = self.posencode(pos, image, src_mask, tgt_mask)
            pos = pos_kl[-1].unsqueeze(1)
            # with torch.no_grad():
            x_start = self.capembed(tgt)
            x_t,t = noise_sample(x_start)
            x_t = self.li2(x_t)
            latent = self.diff_embed(x_t,t)
            latent = latent + pos.expand_as(latent)
            x_t = self.norm_p1(latent)
            out1 = self.decode(image, src_mask,x_t)
            out_pos = self.decoder2(pos.expand_as(latent), image.detach(), src_mask, tgt_mask)
            out_pos = self.generator_pos(out_pos)
            out_pro ,bit_pre= self.generate1(out1)
            tgt_mask = tgt_mask.squeeze(1)
            bitloss = self.loss_bit(x_start, bit_pre, tgt_mask.squeeze(1))
            return bitloss, out_pro, tgt_mask, out_pos, pos_kl[1]
            # x_start = self.capembed(tgt)
        else:
            "Take in and process masked src and target sequences."
            image = self.encode(src, src_mask)
            with torch.no_grad():
                pos_kl = self.posencode(pos, image, src_mask, tgt_mask)
            pos_la = pos_kl[-1].unsqueeze(1) # 这个是隐空间生成的POSz
            # diffusion_based caption decoder 
            x_start = self.capembed(tgt)
            x_t, t = noise_sample(x_start)
            x_t = self.li2(x_t)
            latent = self.diff_embed(x_t, t)
            pos_h = self.posembed(pos)
            latent = latent + pos_h.expand_as(latent)
            x_t = self.norm_p1(latent)
            out1 = self.decode(image, src_mask, x_t)
            out_pos = self.decoder2(pos_la.expand_as(latent), image.detach(), src_mask, tgt_mask)
            out_pro, bit_pre= self.generate1(out1)
            out_pos = self.generator_pos(out_pos)
            tgt_mask = tgt_mask.squeeze(1)
            bitloss = self.loss_bit(x_start, bit_pre, tgt_mask.squeeze(1))
            return bitloss, out_pro, tgt_mask, out_pos, 0

    def encode(self, src, src_mask):
        hidden_state = self.encoder(src,src_mask)
        # hidden_state = self.output_transform(hidden_state)
        # hidden_state = self.output_linear(hidden_state)
        return hidden_state
        # return self.encoder(self.src_embed(src), src_mask)
    def sample11(self, image, img_masks, text_masks=None):
        timesteps = 1000
        batch = image.size(0)
        shape = (batch,20,512)
        noise = torch.randn(*shape, device=image.device)
        indices = list(range(0, timesteps))[::-1]
        indices_next = indices[1:] + [0]
        for i, j in zip(indices, indices_next):
            t = torch.tensor([i] * shape[0], device=image.device)
            next_t = torch.tensor([j] * shape[0], device=image.device)
            with torch.no_grad():
                latent = self.diff_embed(noise, t)
                out_latent = self.decode(image, img_masks,
                                         latent)
                out = self.gs_diff.p_mean_variance(out_latent,
                                                   noise,
                                                   t)
                eps = self.gs_diff._predict_eps_from_xstart(noise, t, out["pred_xstart"])
                alpha_bar = _extract_into_tensor(self.gs_diff.alphas_cumprod, t, noise.shape)
                alpha_bar_prev = _extract_into_tensor(self.gs_diff.alphas_cumprod, next_t, noise.shape)
                sigma = (
                        0.0
                        * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                        * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
                )
                noise = torch.randn_like(noise)
                mean_pred = (
                        out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                        + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
                )
                sample = out["mean"] + torch.exp(0.5 * out["log_variance"]) * noise
                noise = sample
        out_pro = self.generator(noise)
        _, out = torch.max(out_pro, dim=-1)
        return out
    def decode(self, memory, src_mask, tgt, tgt_mask=None):
        hidden_state = self.decoder(tgt,memory,src_mask,tgt_mask)

        # preds = self.classifier(hidden_state.last_hidden_state)
        return hidden_state
        # return self.decoder( tgt,memory, src_mask, tgt_mask)
    def generatorPos(self, src, src_mask):
        with torch.no_grad():
            image = self.encode(src, src_mask)
            out_pos = self.decoder2(pos.expand_as(latent), image.detach(), src_mask, tgt_mask)
            out_pos = self.generator_pos(out_pos)
            return out_pos
        


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









    
class VIDM_DCIC(AttModel):

    def __init__(self, opt):
        super(diff_TransformerModel, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)
        self.stage = 0
        
        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.d_model),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.d_model),) if self.use_bn == 2 else ())))

        delattr(self, 'embed')
        self.embed = lambda x: x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x: x
        delattr(self, 'logit')
        del self.ctx2att
        self.bit_scale = 1.0
        tgt_vocab = self.vocab_size + 1
        self.bit_dim =14
        self.model = self.make_model(0, tgt_vocab,
                                     N_enc=self.N_enc,
                                     N_dec=self.N_dec,
                                     d_model=self.d_model,
                                     d_ff=self.d_ff,
                                     h=self.h,
                                     dropout=self.dropout)
    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        model = EncoderDecoder(
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
                posEncoder(DecoderLayer(d_model, c(attn), c(attn),
                                        c(ff), dropout), N_dec),
                Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                    c(ff), dropout), N_dec),
                Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                    c(ff), dropout), N_dec),
                lambda x: x,
                Embeddings(d_model, 14),
                bit_Embeddings(512, 9488, 14),
                diff_Embeddings(d_model),
                Generator(d_model, 14),
                GaussianDiffusion())
        
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model
    
    def logit(self, x):  # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)
        return memory, att_masks
    
    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            seq_mask = (seq.data != 0) & (seq.data != -1)
            seq_mask[:, 0] = 1  # bos
            seq_mask = torch.cat([seq_mask.new(seq_mask.size(0), 1).fill_(1), seq_mask[:, :-1]], 1)
            seq_mask = seq_mask.unsqueeze(-2)
            # seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                                                            [att_feats, att_masks]
                                                            )
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask
    
    def generate(self, out):
        outputs1 = self.model.generator(out)
        buffer_probs = nn.Softmax(dim=-1)(outputs1)
        vocab_inds = torch.arange(0, self.vocab_size + 1).long().view(1, self.vocab_size + 1, 1, 1).cuda()
        vocab_bit_buffer = decimal_to_bits(vocab_inds, vocab_size=self.vocab_size + 1, bits=self.bit_dim) * self.bit_scale
        logits_w = torch.matmul(buffer_probs, vocab_bit_buffer.expand(buffer_probs.size(0), -1, -1))
        return outputs1
    
    def _forward(self, fc_feats, att_feats, pos, seq, att_masks=None):
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        return(self.model(att_feats,  pos,seq, att_masks, seq_mask, self.stage))

    def core(self, it, pos, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out, _, pos_g = self.model.decode(memory, pos, mask,
                                          ys,
                                          subsequent_mask(ys.size(1))
                                          .to(memory.device), False)
        return out[:, -1], [ys.unsqueeze(0)], pos_g
    def generator_pos(self, att_feat, att_masks):
        return self.model.generatorPos(att_feat, att_masks)

'''
gaussian KL loss fucntion
'''
def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                           - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                           - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), dim=-1)
    return kld


# no use
class CodeBook(nn.Module):
    def __init__(self, cfg):
        super(CodeBook, self).__init__()
        self.embedding = nn.Embedding(cfg.num_modes,20, cfg.hidden_size)
        self.commitment_cost = cfg.loss.commitment_cost

    def forward(self, mode_emb, splits):
        distances = (torch.sum(mode_emb**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(mode_emb, self.embedding.weight.t())).sqrt()

        distances = distances.split(splits)
        indices = [
            linear_sum_assignment(d.detach().cpu().numpy())[1]
            for d in distances
        ]
        indices = torch.from_numpy(np.concatenate(indices))
        indices = indices.to(mode_emb.device)
        # print("indices: ", indices.unique())
        quantized = self.embedding(indices)

		# Loss
        q_latent_loss = F.mse_loss(mode_emb.detach(), quantized) + \
                        F.mse_loss(mode_emb.mean(dim=0).detach(),
                                   self.embedding.weight.mean(dim=0))
        e_latent_loss = F.mse_loss(mode_emb, quantized.detach()) + \
                        F.mse_loss(mode_emb.mean(dim=0),
                                   self.embedding.weight.mean(dim=0).detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = mode_emb + (quantized - mode_emb).detach()

        return loss, quantized[:, None, :], indices
    
class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim=256):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered