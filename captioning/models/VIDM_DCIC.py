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

class VIDM_DCIC(AttModel):

    def __init__(self, opt):
        super(diff_TransformerModel, self).__init__(opt)
        self.opt = opt
        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)
        self.stage = 0
        # delattr 函数用于删除属性。 delattr(x, 'foobar') 相等于 del x.foobar。
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

