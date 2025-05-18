import math
import time
from functools import lru_cache

import constriction
import numpy as np
import torch
import torch.nn as nn

from layer.layer_utils import quantize_ste, make_conv
from layer.transformer import TransformerBlock, SwinTransformerBlock
from net.elic import ELICAnalysis, ELICSynthesis


class GaussianMixtureEntropyModel(nn.Module):
    # Todo: Re-implement with Compressai for faster inference speed
    def __init__(
            self,
            minmax: int = 64
    ):
        super().__init__()
        self.minmax = minmax
        self.samples = torch.arange(-minmax, minmax + 1, 1, dtype=torch.float32).cuda()
        self.laplace = torch.distributions.Laplace(0, 1)
        self.pmf_laplace = self.laplace.cdf(self.samples + 0.5) - self.laplace.cdf(self.samples - 0.5)
        # self.gaussian_conditional = GaussianConditional(None)

    def update_minmax(self, minmax):
        self.minmax = minmax
        self.samples = torch.arange(-minmax, minmax + 1, 1, dtype=torch.float32).cuda()
        self.pmf_laplace = self.laplace.cdf(self.samples + 0.5) - self.laplace.cdf(self.samples - 0.5)

    def get_likelihood(self, latent_hat, probs, means, scales):
        gaussian1 = torch.distributions.Normal(means[0], scales[0])
        gaussian2 = torch.distributions.Normal(means[1], scales[1])
        gaussian3 = torch.distributions.Normal(means[2], scales[2])
        likelihoods_0 = gaussian1.cdf(latent_hat + 0.5) - gaussian1.cdf(latent_hat - 0.5)
        likelihoods_1 = gaussian2.cdf(latent_hat + 0.5) - gaussian2.cdf(latent_hat - 0.5)
        likelihoods_2 = gaussian3.cdf(latent_hat + 0.5) - gaussian3.cdf(latent_hat - 0.5)

        likelihoods = 0.999 * (probs[0] * likelihoods_0 + probs[1] * likelihoods_1 + probs[2] * likelihoods_2)
        + 0.001 * (self.laplace.cdf(latent_hat + 0.5) - self.laplace.cdf(latent_hat - 0.5))
        likelihoods = likelihoods + 1e-10
        return likelihoods

    def get_GMM_pmf(self, probs, means, scales):
        L = self.samples.size(0)
        num_symbol = probs.size(1)
        samples = self.samples.unsqueeze(0).repeat(num_symbol, 1)  # N 65
        scales = scales.unsqueeze(-1).repeat(1, 1, L)
        means = means.unsqueeze(-1).repeat(1, 1, L)
        probs = probs.unsqueeze(-1).repeat(1, 1, L)
        likelihoods_0 = self._likelihood(samples, scales[0], means=means[0])
        likelihoods_1 = self._likelihood(samples, scales[1], means=means[1])
        likelihoods_2 = self._likelihood(samples, scales[2], means=means[2])
        pmf_clip = (0.999 * (probs[0] * likelihoods_0 + probs[1] * likelihoods_1 + probs[2] * likelihoods_2)
                    + 0.001 * self.pmf_laplace)
        return pmf_clip

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def compress(self, symbols, probs, means, scales):
        pmf_clip = self.get_GMM_pmf(probs, means, scales)
        model_family = constriction.stream.model.Categorical()  # note empty `()`
        probabilities = pmf_clip.cpu().numpy().astype(np.float64)
        symbols = symbols.reshape(-1)
        symbols = (symbols + self.minmax).cpu().numpy().astype(np.int32)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(symbols, model_family, probabilities)
        compressed = encoder.get_compressed()
        return compressed

    def decompress(self, compressed, probs, means, scales):
        pmf = self.get_GMM_pmf(probs, means, scales).cpu().numpy().astype(np.float64)
        model = constriction.stream.model.Categorical()
        decoder = constriction.stream.queue.RangeDecoder(compressed)
        symbols = decoder.decode(model, pmf)
        symbols = torch.from_numpy(symbols).to(probs.device) - self.minmax
        symbols = torch.tensor(symbols, dtype=torch.float32)
        return symbols


@lru_cache()
def get_qlds(H, W, device):
    n, m, g = 0, 0, 1.32471795724474602596
    a1, a2 = 1.0 / g, 1.0 / g / g
    context_tensor = torch.zeros((H, W)).to(device) - 1
    while m < H * W:
        n += 1
        x = int(round(((0.5 + n * a1) % 1) * H)) % H
        y = int(round(((0.5 + n * a2) % 1) * W)) % W
        if context_tensor[x, y] == -1:
            context_tensor[x, y] = m
            m += 1
    return context_tensor


@lru_cache()
def get_coding_order(target_shape, context_mode, device, step=12, beta=2.2):
    if device == -1:
        device = 'cpu'
    if context_mode == 'quincunx':
        context_tensor = torch.tensor([[4, 2, 4, 0], [3, 4, 3, 4], [4, 1, 4, 2]]).to(device)
    elif context_mode == 'checkerboard2':
        context_tensor = torch.tensor([[1, 0], [0, 1]]).to(device)
    elif context_mode == 'checkerboard4':
        context_tensor = torch.tensor([[0, 2], [3, 1]]).to(device)
    elif context_mode == 'qlds':
        B, C, H, W = target_shape

        def get_qlds(H, W):
            n, m, g = 0, 0, 1.32471795724474602596
            a1, a2 = 1.0 / g, 1.0 / g / g
            context_tensor = torch.zeros((H, W)).to(device) - 1
            while m < H * W:
                n += 1
                x = int(round(((0.5 + n * a1) % 1) * H)) % H
                y = int(round(((0.5 + n * a2) % 1) * W)) % W
                if context_tensor[x, y] == -1:
                    context_tensor[x, y] = m
                    m += 1
            return context_tensor

        context_tensor = torch.tensor(get_qlds(H, W), dtype=torch.int)

        def gamma_func(beta=1.):
            return lambda r: r ** beta

        ratio = 1. * (np.arange(step) + 1) / step
        gamma = gamma_func(beta=beta)
        L = H * W  # total number of tokens
        mask_ratio = np.clip(np.floor(L * gamma(ratio)), 0, L - 1)
        for i in range(step):
            context_tensor = torch.where((context_tensor <= mask_ratio[i]) * (context_tensor > i),
                                         torch.ones_like(context_tensor) * i, context_tensor)
        return context_tensor
    else:
        context_tensor = context_mode
    B, C, H, W = target_shape
    Hp, Wp = context_tensor.size()
    coding_order = torch.tile(context_tensor, (H // Hp + 1, W // Wp + 1))[:H, :W]
    return coding_order


@lru_cache()
def get_resicomp_order(target_shape, device, ratio_base=0.05, step=12, alpha=2.2):
    B, C, H, W = target_shape

    def get_qlds(H, W):
        n, m, g = 0, 0, 1.32471795724474602596
        a1, a2 = 1.0 / g, 1.0 / g / g
        context_tensor = torch.zeros((H, W)).to(device) - 1
        while m < H * W:
            n += 1
            x = int(round(((0.5 + n * a1) % 1) * H)) % H
            y = int(round(((0.5 + n * a2) % 1) * W)) % W
            if context_tensor[x, y] == -1:
                context_tensor[x, y] = m
                m += 1
        return context_tensor

    context_tensor = torch.tensor(get_qlds(H, W), dtype=torch.int)

    def gamma_func(alpha=1.):
        return lambda r: r ** alpha

    gamma = gamma_func(alpha=alpha)
    ratio = ratio_base * gamma(1. * (np.arange(step) + 1) / step)
    ratio = np.asarray(ratio.tolist() + [1.])
    L = H * W  # total number of tokens
    mask_ratio = np.clip(np.floor(L * ratio), 0, L - 1)
    for i in range(step + 1):
        context_tensor = torch.where((context_tensor <= mask_ratio[i]) * (context_tensor > i),
                                     torch.ones_like(context_tensor) * i, context_tensor)
    return context_tensor


@lru_cache()
def get_context_mode(packet_num, context_mode):
    if context_mode == 'Layered':  # for layered coding
        context_matrix = np.tril(np.ones((packet_num, packet_num)), k=-1)
    elif context_mode == 'IntraSlice':
        context_matrix = np.zeros((packet_num, packet_num))
    elif context_mode == 'MultiDescription_2':
        context_matrix = np.zeros((packet_num, packet_num))
        mask = np.tril(np.ones((packet_num, packet_num)), k=-2)
        context_matrix[2::2, 0::2] = 1
        context_matrix[3::2, 1::2] = 1
        context_matrix = context_matrix * mask
    elif context_mode == 'MultiDescription_3':
        context_matrix = np.zeros((packet_num, packet_num))
        mask = np.tril(np.ones((packet_num, packet_num)), k=-3)
        context_matrix[3::3, 0::3] = 1
        context_matrix[4::3, 1::3] = 1
        context_matrix[5::3, 2::3] = 1
        context_matrix = context_matrix * mask
    elif context_mode == 'MultiDescription_4':
        context_matrix = np.zeros((packet_num, packet_num))
        mask = np.tril(np.ones((packet_num, packet_num)), k=-4)
        context_matrix[4::4, 0::4] = 1
        context_matrix[5::4, 1::4] = 1
        context_matrix[6::4, 2::4] = 1
        context_matrix[7::4, 3::4] = 1
        context_matrix = context_matrix * mask
    elif context_mode == 'MultiDescription_5':
        context_matrix = np.zeros((packet_num, packet_num))
        mask = np.tril(np.ones((packet_num, packet_num)), k=-5)
        context_matrix[5::5, 0::5] = 1
        context_matrix[6::5, 1::5] = 1
        context_matrix[7::5, 2::5] = 1
        context_matrix[8::5, 3::5] = 1
        context_matrix[9::5, 4::5] = 1
        context_matrix = context_matrix * mask
    elif context_mode == 'LayeredMDC':
        context_matrix = np.zeros((packet_num, packet_num))
        mask = np.tril(np.ones((packet_num, packet_num)), k=-1)
        context_matrix[1:, 0::2] = 1
        context_matrix = context_matrix * mask
        # print(dependency_structure)
    elif context_mode == 'SLC':
        context_matrix = np.zeros((packet_num, packet_num))
        mask = np.tril(np.ones((packet_num, packet_num)), k=-1)
        context_matrix[1:, 0] = 1
        context_matrix = context_matrix * mask
        # print(dependency_structure)
    elif context_mode == 'TwoLoop':
        context_matrix = np.zeros((packet_num, packet_num))
        mask = np.tril(np.ones((packet_num, packet_num)), k=-1)
        context_matrix[1:, 0::2] = 1
        context_matrix[3::2, 1::2] = 1
        context_matrix = context_matrix * mask
        # print(dependency_structure)
    else:
        print('Not implemented')
        raise NotImplementedError
    return context_matrix


class DualFunctionalTransformer(nn.Module):
    def __init__(self, latent_dim, dim=768, depth=12, num_heads=12, window_size=24,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, transformer='swin'):
        super().__init__()
        self.dim = dim
        self.depth = depth
        if transformer == 'swin':
            window_size = 4
            num_heads = 8
            self.blocks = nn.ModuleList([
                SwinTransformerBlock(dim=dim,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
                for i in range(depth)])
        self.delta = 5.0
        self.embedding_layer = nn.Linear(latent_dim, dim)
        # self.positional_encoding = LearnedPosition(window_size * window_size, dim)
        self.entropy_parameters = nn.Sequential(
            make_conv(dim, dim * 4, 1, 1),
            nn.GELU(),
            make_conv(dim * 4, dim * 4, 1, 1),
            nn.GELU(),
            make_conv(dim * 4, latent_dim * 9, 1, 1),
        )

        self.prediction_head = nn.Sequential(
            make_conv(dim, dim * 4, 1, 1),
            nn.GELU(),
            make_conv(dim * 4, dim * 4, 1, 1),
            nn.GELU(),
            make_conv(dim * 4, latent_dim, 1, 1),
        )

        self.gmm_model = GaussianMixtureEntropyModel()
        self.laplace = torch.distributions.Laplace(0, 1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim), requires_grad=True)

    def forward_with_given_mask(self, latent_hat, mask, slice_size=None):
        B, C, H, W = latent_hat.size()
        device = latent_hat.get_device()
        input_resolution = (H, W)
        x = latent_hat.flatten(2).transpose(1, 2)  # B L N
        mask_BLN = mask.flatten(2).transpose(1, 2)  # B L N
        x_masked = x * mask_BLN + self.mask_token * (1 - mask_BLN)

        x_masked = self.embedding_layer(x_masked / self.delta)
        # x = self.positional_encoding(x)
        for _, blk in enumerate(self.blocks):
            x_masked = blk(x_masked, input_resolution, device, slice_size)
        x_out = x_masked.transpose(1, 2).reshape(B, self.dim, H, W)
        params = self.entropy_parameters(x_out)
        probs, means, scales = params.chunk(3, dim=1)
        probs = torch.softmax(probs.reshape(B, 3, C, H, W), dim=1).transpose(0, 1)
        means = means.reshape(B, 3, C, H, W).transpose(0, 1)
        scales = torch.abs(scales).reshape(B, 3, C, H, W).transpose(0, 1).clamp(1e-10, 1e10)
        latent_pred = self.prediction_head(x_out)
        return probs, means, scales, latent_pred

    def forward_with_random_mask(self, latent, masked_ratio=None):
        B, C, H, W = latent.size()
        half = float(0.5)
        noise = torch.empty_like(latent).uniform_(-half, half)
        latent_noise = latent + noise
        latent_hat = quantize_ste(latent)

        def generate_random_mask(latent, r):
            mask_loc = torch.randn(H * W).to(latent.get_device())
            threshold = torch.sort(mask_loc)[0][r]
            mask = torch.where(mask_loc >= threshold, torch.ones_like(mask_loc), torch.zeros_like(mask_loc))
            mask = mask.reshape(1, 1, H, W).repeat(B, C, 1, 1)
            return mask

        if masked_ratio is not None:
            r = int(masked_ratio * H * W)
        else:
            r = math.floor(np.random.uniform(0.05, 0.99) * H * W)  # drop probability
        mask = generate_random_mask(latent_hat, r)
        mask_params = mask.unsqueeze(0).repeat(3, 1, 1, 1, 1)
        probs, means, scales, latent_pred = self.forward_with_given_mask(latent_hat, mask)
        likelihoods_masked = torch.ones_like(latent_hat)
        likelihoods = self.gmm_model.get_likelihood(latent_noise[mask == 0],
                                                    probs[mask_params == 0].reshape(3, -1),
                                                    means[mask_params == 0].reshape(3, -1),
                                                    scales[mask_params == 0].reshape(3, -1))
        likelihoods_masked[mask == 0] = likelihoods

        latent_pred = latent_hat * mask + latent_pred * (1 - mask)
        return latent_hat, likelihoods_masked, latent_pred, mask

    def encode_latents(self, latent_hat, ctx_locations, encoding_locations):
        device = latent_hat.get_device()
        mask_i = torch.where(ctx_locations == 1., torch.ones_like(latent_hat), torch.zeros_like(latent_hat))
        probs_i, means_i, scales_i, latent_pred = self.forward_with_given_mask(latent_hat, mask_i)
        mask_params_i = encoding_locations.unsqueeze(0).repeat(3, 1, 1, 1, 1)
        likelihoods_i = self.gmm_model.get_likelihood(latent_hat[encoding_locations],
                                                      probs_i[mask_params_i].reshape(3, -1),
                                                      means_i[mask_params_i].reshape(3, -1),
                                                      scales_i[mask_params_i].reshape(3, -1))
        bpp_i = - torch.log2(likelihoods_i).sum().item() / 512 / 768
        return likelihoods_i, bpp_i, latent_pred

    def inference(self, latent_hat, step=2, beta=2.2, context_mode='qlds', slice_size=None):
        coding_order = get_coding_order(latent_hat.shape, context_mode, latent_hat.get_device(), step=step,
                                        beta=beta)  # H W
        # coding_order = get_resicomp_order(latent_hat.shape, latent_hat.get_device(), ratio_base=0.02, step=step, alpha=alpha)  # H W
        coding_order = coding_order.reshape(1, 1, *coding_order.shape).repeat(latent_hat.shape[0], latent_hat.shape[1],
                                                                              1, 1)
        total_steps = int(coding_order.max() + 1)
        likelihoods = torch.zeros_like(latent_hat)
        bpp = 0
        for i in range(total_steps):
            ctx_locations = (coding_order < i)
            encoding_locations = (coding_order == i)
            likelihoods_i, bpp_i, _ = self.encode_latents(latent_hat, ctx_locations, encoding_locations)
            likelihoods[encoding_locations] = likelihoods_i
            bpp = bpp + bpp_i
        return likelihoods

    def progressive_inference(self, latent_hat, step=12):
        coding_order = get_coding_order(latent_hat.shape, 'qlds', latent_hat.get_device(), step=step)  # H W
        coding_order = coding_order.reshape(1, 1, *coding_order.shape).repeat(latent_hat.shape[0], latent_hat.shape[1],
                                                                              1, 1)
        total_steps = int(coding_order.max() + 1)
        likelihoods = torch.ones_like(latent_hat)
        progressive_likelihoods = []
        progressive_latents = []
        for i in range(total_steps):
            ctx_locations = (coding_order < i)
            encoding_locations = (coding_order == i)
            mask_i = torch.where(ctx_locations == 1., torch.ones_like(latent_hat), torch.zeros_like(latent_hat))
            likelihoods_i, bpp_i, latent_pred = self.encode_latents(latent_hat, ctx_locations, encoding_locations)
            likelihoods[encoding_locations] = likelihoods_i
            progressive_likelihoods.append(likelihoods.clone())
            if i != 0:
                latent_pred = latent_hat * mask_i + latent_pred * (1 - mask_i)
                progressive_latents.append(latent_pred.clone())
        progressive_latents.append(latent_hat.clone())
        return progressive_likelihoods, progressive_latents

    def encode_latents_with_multistep(self, latent_hat, ctx_locations, encoding_locations, total_steps=2):
        # support intra-slice contextual encoding with multiple steps for better RD performance
        B, C, H, W = latent_hat.size()
        device = latent_hat.get_device()
        random_locations = torch.rand(1, 1, H, W).to(device)
        multistep_encoding_locations = torch.ceil(random_locations * total_steps) * encoding_locations
        likelihoods = torch.zeros_like(latent_hat)
        for i in range(total_steps):
            encoding_locations_i = (multistep_encoding_locations == i + 1)
            likelihoods_i, bpp_i, _ = self.encode_latents(latent_hat, ctx_locations, encoding_locations_i)
            ctx_locations = ctx_locations + encoding_locations_i
            likelihoods[encoding_locations_i] = likelihoods_i
        return likelihoods

    def encode_latents_into_packets(self, latent_hat, context_mode, beta, packet_num=10, per_packet_step=1):
        B, C, H, W = latent_hat.size()
        context_matrix = get_context_mode(packet_num, context_mode)
        # partition the latent space into multiple packets
        coding_order = get_qlds(H, W, latent_hat.get_device())  # H W
        C_i = (context_matrix.sum(axis=1) / packet_num + 1) ** beta
        C_sum = C_i.sum()
        ratio = C_i / C_sum
        L = H * W  # total number of tokens
        mask_ratio = np.clip(np.floor(L * ratio.cumsum()), 0, L - 1)
        for i in range(packet_num):
            coding_order = torch.where((coding_order <= mask_ratio[i]) * (coding_order > i),
                                       torch.ones_like(coding_order) * i, coding_order)
        coding_order = coding_order.reshape(1, 1, *coding_order.shape).repeat(B, C, 1, 1)
        packet_loc = []  # indicate the token location of each packet
        for i in range(packet_num):
            packet_loc_i = (coding_order == i)  # BxCxHxW
            packet_loc.append(packet_loc_i)

        likelihoods = torch.zeros_like(latent_hat)
        # packet-wise encoding
        bpp_list = np.zeros(packet_num)
        Nl_list = np.zeros(packet_num)  # number of latents in each slice
        for i in range(packet_num):
            ctx_locations = torch.zeros_like(latent_hat)
            for j in range(i):
                if context_matrix[i, j] == 1:  # slice based context modeling
                    ctx_locations = ctx_locations + packet_loc[j]
            likelihoods_i = self.encode_latents_with_multistep(latent_hat, ctx_locations, packet_loc[i],
                                                               4 if i == 0 else per_packet_step)
            bpp_i = - torch.log2(likelihoods_i[packet_loc[i]]).sum().item() / L / 256
            bpp_list[i] = bpp_i
            Nl_list[i] = torch.sum(packet_loc[i][0, 0]).item()
            likelihoods = likelihoods + likelihoods_i
            # print('packet_id {}, bpp={}, num_latents={}'.format(i, bpp_i, Nl_list[i]))
        return likelihoods, bpp_list, Nl_list, packet_loc, context_matrix

    # def latent_plc(self, latent_hat, effective_packet_ids, beta, packet_num=10):
    #     B, C, H, W = latent_hat.size()
    #     # partition the latent space into multiple packets
    #     coding_order = get_coding_order(latent_hat.shape, 'qlds', latent_hat.get_device(),
    #                                     step=packet_num, beta=beta)  # H W
    #     coding_order = coding_order.reshape(1, 1, *coding_order.shape).repeat(B, C, 1, 1)
    #     mask = torch.zeros_like(latent_hat)
    #     for i in effective_packet_ids:
    #         mask[coding_order == i] = 1
    #     _, _, _, latent_pred = self.forward_with_given_mask(latent_hat, mask)
    #     latent_pred = latent_hat * mask + latent_pred * (1 - mask)
    #     return latent_pred

    def compress(self, latent, context_mode='qlds'):
        latent_hat = torch.round(latent)
        self.gmm_model.update_minmax(int(latent_hat.max().item()))
        coding_order = get_coding_order(latent.shape, context_mode, latent_hat.get_device(), step=12)  # H W
        coding_order = coding_order.reshape(1, 1, *coding_order.shape).repeat(latent.shape[0], latent.shape[1], 1, 1)
        total_steps = int(coding_order.max() + 1)
        strings = []
        for i in range(total_steps):
            ctx_locations = (coding_order < i)
            encoding_locations = (coding_order == i)
            mask_params_i = encoding_locations.unsqueeze(0).repeat(3, 1, 1, 1, 1)
            mask_i = torch.where(ctx_locations, torch.ones_like(latent), torch.zeros_like(latent))
            probs_i, means_i, scales_i, _ = self.forward_with_given_mask(latent_hat, mask_i)
            string_i = self.gmm_model.compress(latent_hat[encoding_locations],
                                               probs_i[mask_params_i].reshape(3, -1),
                                               means_i[mask_params_i].reshape(3, -1),
                                               scales_i[mask_params_i].reshape(3, -1))
            strings.append(string_i)
        return strings

    def decompress(self, strings, latent_size, device, context_mode='qlds'):
        B, C, H, W = latent_size
        coding_order = get_coding_order(latent_size, context_mode, device, step=12)  # H W
        coding_order = coding_order.reshape(1, 1, *coding_order.shape).repeat(B, C, 1, 1)
        total_steps = int(coding_order.max() + 1)
        t0 = time.time()
        latent_hat = torch.zeros(latent_size).to(device)
        for i in range(total_steps):
            ctx_locations = (coding_order < i)
            encoding_locations = (coding_order == i)
            mask_params_i = encoding_locations.unsqueeze(0).repeat(3, 1, 1, 1, 1)
            mask_i = torch.where(ctx_locations, torch.ones_like(latent_hat), torch.zeros_like(latent_hat))
            probs_i, means_i, scales_i, _ = self.forward_with_given_mask(latent_hat, mask_i)
            symbols_i = self.gmm_model.decompress(strings[i],
                                                  probs_i[mask_params_i].reshape(3, -1),
                                                  means_i[mask_params_i].reshape(3, -1),
                                                  scales_i[mask_params_i].reshape(3, -1))
            latent_hat[encoding_locations] = symbols_i
        print('decompress', time.time() - t0)
        return latent_hat

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mask_token'}


class ResiComp(nn.Module):
    def __init__(self):
        super().__init__()
        self.g_a = ELICAnalysis()
        self.g_s = ELICSynthesis()
        self.mim = DualFunctionalTransformer(192)

    def forward(self, x):
        y = self.g_a(x)
        y_hat, likelihoods, y_pred, _ = self.mim.forward_with_random_mask(y)
        x_hat = self.g_s(y_hat)
        x_check = self.g_s(y_pred)
        return {
            "x_hat": x_hat,
            "x_check": x_check,
            "likelihoods": likelihoods,
        }

    def forward_with_random_mask(self, x, masked_ratio):
        y = self.g_a(x)
        y_hat, likelihoods, y_pred, mask = self.mim.forward_with_random_mask(y, masked_ratio)
        x_hat = self.g_s(y_hat)
        x_check = self.g_s(y_pred)
        return {
            "x_hat": x_hat,
            "x_check": x_check,
            "mask": mask,
            "likelihoods": likelihoods,
        }

    def inference_with_packet_loss(self, x, packet_tracer, context_mode, N=100, step=10, beta=2.2):
        packet_num = step
        y = self.g_a(x)
        y_hat = quantize_ste(y)
        likelihoods, bpp_list, Nl_list, packet_loc, context_matrix = (
            self.mim.encode_latents_into_packets(y_hat, context_mode, beta, per_packet_step=4))
        psnr_list = []
        for iter in range(N):
            packet_loss_idx = packet_tracer.generate(packet_num)
            effective_packet_idx = ''

            # get the effective packet ids
            for ind, t in enumerate(packet_loss_idx):
                if t == '0':
                    effective_packet_idx += '0'
                elif t == '1':
                    # check if the previous context packets are all received
                    flag = True
                    for k in range(ind):
                        if context_matrix[ind, k] == 1:
                            if effective_packet_idx[k] == '0':
                                flag = False
                                break
                    effective_packet_idx += '1' if flag else '0'
                else:
                    raise ValueError("Invalid packet loss index")
            # collect the effective packet ids
            effective_packet_ids = []
            for ind, t in enumerate(effective_packet_idx):
                if t == '1':
                    effective_packet_ids.append(ind)

            # result = self.resilient_decoding(y_hat, effective_packet_ids, beta, packet_num)

            # one can save the mapping from effective_packet_ids->psnr
            mask = torch.zeros_like(y_hat)
            for i in effective_packet_ids:
                mask[packet_loc[i]] = 1
            _, _, _, y_pred = self.mim.forward_with_given_mask(y_hat, mask)
            y_pred = y_hat * mask + y_pred * (1 - mask)
            x_check = self.g_s(y_pred)

            psnr = 10 * torch.log10(1 / torch.mean((x - x_check) ** 2))
            psnr_list.append(psnr.item())
            print(
                'Iter {} | packet_loss_idx {} | effective_packet_idx {} | effective_packet_ids {} | psnr {}'.format(
                    iter + 1, packet_loss_idx, effective_packet_idx, effective_packet_ids, psnr.item()))
        avg_psnr = sum(psnr_list) / len(psnr_list)
        bpp = sum(bpp_list)
        return avg_psnr, bpp

    def inference_without_packet_loss(self, x, step=12, beta=2.2):
        y = self.g_a(x)
        y_hat = torch.round(y)
        likelihoods = self.mim.inference(y_hat, step=step, beta=beta)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": likelihoods,
        }

    def real_inference_without_packet_loss(self, x):
        num_pixels = x.size(2) * x.size(3)
        y = self.g_a(x)
        strings = self.mim.compress(y)
        y_hat = self.mim.decompress(strings, y.shape, x.get_device())
        x_hat = self.g_s(y_hat)
        bpp = sum([string.size * 32 for string in strings]) / num_pixels
        return {
            "x_hat": x_hat,
            "bpp": bpp,
        }

    def inference_for_progressive_decoding(self, x, step=64):
        y = self.g_a(x)
        y_hat = quantize_ste(y)
        likelihoods, latent_pred = self.mim.progressive_inference(y_hat, step=step)
        x_check_list = []
        for i in range(step):
            x_check = self.g_s(latent_pred[i])
            x_check_list.append(x_check)
        return {
            "x_check_list": x_check_list,
            "likelihoods_list": likelihoods
        }

    # def packet_level_inference(self, x, context_mode, beta, per_packet_step=2):
    #     y = self.g_a(x)
    #     y_hat = quantize_ste(y)
    #     likelihoods, bpp_list, Nl_list, packet_loc, context_matrix = (
    #         self.mim.encode_latents_into_packets(y_hat, context_mode, beta, per_packet_step=per_packet_step))
    #     x_hat = self.g_s(y_hat)
    #     return {
    #         "x_hat": x_hat,
    #         "likelihoods": likelihoods,
    #         "bpp_list": bpp_list,
    #         "Nl_list": Nl_list,
    #     }

    def tokenization(self, x):
        y = self.g_a(x)
        y_hat = quantize_ste(y)
        return y_hat

    # def resilient_decoding(self, y_hat, effective_packet_ids, beta, packet_num=10):
    #     y_pred = self.mim.latent_plc(y_hat, effective_packet_ids, beta, packet_num)
    #     x_check = self.g_s(y_pred)
    #     return {
    #         "x_check": x_check.clamp(0, 1),
    #     }
