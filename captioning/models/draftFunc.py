
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