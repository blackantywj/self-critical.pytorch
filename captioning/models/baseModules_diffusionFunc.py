from .captioning.utils.utils import *
def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))
def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))
def noise_sample(inputs):
    # corrupt bit repr, i.e. 14-dim vector
    batch_size = inputs.shape[0]
    device = inputs.device

    # sample random times
    times = torch.zeros((batch_size,), device = device).float().uniform_(0, 0.999)

    bit_token_embed = inputs
    noise = torch.randn_like(bit_token_embed)

    noise_level = alpha_cosine_log_snr(times)
    padded_noise_level = right_pad_dims_to(bit_token_embed, noise_level)
    alpha, sigma =  log_snr_to_alpha_sigma(padded_noise_level) # 从 noise 那里去生成 alpha 和均值量

    noised_bit_token_embed = alpha * bit_token_embed + sigma * noise
    return {
        noised_bit_token_embed,
        noise_level
    }
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))
def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5)
def decimal_to_bits(x,vocab_size, bits):
    """ expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1 """
    device = x.device

    x = x.clamp(0, vocab_size - 1)

    mask = 2 ** torch.arange(bits - 1, -1, -1, device=device)
    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b c h w -> b c 1 h w')

    bits = ((x & mask) != 0).float()
    # bits = rearrange(bits, 'b c d h w -> b (c d) h w')
    bits = bits.squeeze(-1).squeeze(-1)  # batch_size x seq_length x bits x 1 x 1 -> batch_size x seq_length x bits
    bits = bits * 2 - 1
    return bits
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
def sample(batch_size, device):

    w = np.ones([1000])
    p = w / np.sum(w)
    indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
    indices = torch.from_numpy(indices_np).long().to(device)
    weights_np = 1 / (len(p) * p[indices_np])
    weights = torch.from_numpy(weights_np).float().to(device)
    return indices