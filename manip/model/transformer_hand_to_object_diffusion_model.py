import os 
import math 

from tqdm.auto import tqdm

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from inspect import isfunction

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import pytorch3d.transforms as transforms 

from manip.model.transformer_module import Decoder 

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_input_feats,  # Input dimension (18 for hand poses)
        d_feats,        # Output dimension (9 for object motion)
        d_model,        # Transformer model dimension
        n_dec_layers,   # Number of decoder layers
        n_head,         # Number of attention heads
        d_k,           # Key dimension
        d_v,           # Value dimension
        max_timesteps,  # Maximum sequence length
    ):
        super().__init__()
        
        self.d_feats = d_feats 
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k 
        self.d_v = d_v 
        self.max_timesteps = max_timesteps 

        # Input: BS X D X T 
        # Output: BS X T X D'
        self.motion_transformer = Decoder(
            d_feats=d_input_feats + self.d_feats,
            d_model=self.d_model,
            n_layers=self.n_dec_layers, 
            n_head=self.n_head, 
            d_k=self.d_k, 
            d_v=self.d_v,
            max_timesteps=self.max_timesteps, 
            use_full_attention=True
        )  

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

        # For noise level t embedding
        dim = 64
        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )

    def forward(self, src, noise_t, condition, padding_mask=None):
        # src: BS X T X D (noisy object motion)
        # condition: BS X T X D_cond (hand poses)
        # noise_t: int (timestep)

        # Use only hand poses as condition
        # src = condition
       
        noise_t_embed = self.time_mlp(noise_t) # BS X d_model 
        noise_t_embed = noise_t_embed[:, None, :] # BS X 1 X d_model 

        bs = src.shape[0]
        num_steps = src.shape[1] + 1

        if padding_mask is None:
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool()

        pos_vec = torch.arange(num_steps)+1
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1)

        data_input = torch.cat([src.transpose(1, 2), condition.transpose(1, 2)], dim=1)
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, obj_embedding=noise_t_embed)
        
        output = self.linear_out(feat_pred[:, 1:])

        return output

class CondGaussianDiffusion(nn.Module):
    def __init__(
        self,
        opt,
        d_feats,        # Output dimension (9 for object motion)
        d_model,        # Transformer model dimension
        n_head,         # Number of attention heads
        n_dec_layers,   # Number of decoder layers
        d_k,           # Key dimension
        d_v,           # Value dimension
        max_timesteps,  # Maximum sequence length
        out_dim,       # Output dimension
        d_input_feats=18,  # Input dimension (18 for 9D hands, 24 for 12D hands)
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_x0',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0.,
        p2_loss_weight_k = 1,
        batch_size=None,
        guidance_weight = 0.0,
        guidance_mode = False,
    ):
        super().__init__()
        self.guidance_weight = guidance_weight
        self.guidance_mode = guidance_mode
        self.clip_denoised = True

        # Use the passed d_input_feats instead of hardcoding
        self.d_input_feats = d_input_feats
            
        self.denoise_fn = TransformerDiffusionModel(
            d_input_feats=d_input_feats,
            d_feats=d_feats,
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            n_dec_layers=n_dec_layers,
            max_timesteps=max_timesteps
        )
        
        self.objective = objective
        self.seq_len = max_timesteps - 1 
        self.out_dim = out_dim 

        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, cond, t, weight=None, clip_x_start=False, nn_cond=None):
        model_output = self.denoise_fn(x, t, cond, None)
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            if clip_x_start:
                x_start.clamp_(-1., 1.)

        elif self.objective == 'pred_x0':
            x_start = model_output
            if clip_x_start:
                x_start.clamp_(-1., 1.)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, x, t, x_cond, padding_mask, clip_denoised):
        pred_noise, x_start = self.model_predictions(x, x_cond, t, clip_x_start=clip_denoised)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, x_cond, padding_mask=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, _ = self.p_mean_variance(x=x, t=t, x_cond=x_cond,             padding_mask=padding_mask, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def long_ddim_sample(self, shape, cond, **model_kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            50,
            1,
        )

        if batch == 1:
            return self.ddim_sample(shape, cond)

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights = np.clip(
            np.linspace(0, self.guidance_weight * 2, sampling_timesteps),
            None,
            self.guidance_weight,
        )
        time_pairs = list(
            zip(times[:-1], times[1:], weights)
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)
        cond = cond.to(device)

        assert batch > 1
        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2

        x_start = None

        for time, time_next, weight in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                x, cond, time_cond, weight=weight, clip_x_start=self.clip_denoised, nn_cond=model_kwargs.get('nn_cond', None)
            )

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(x)

            model_mean = x_start * alpha_next.sqrt() + c * pred_noise

            if self.guidance_mode:
                print('guide')
                # spatial guidance/classifier guidance
                time_next_cond = torch.full((batch,), time_next, device=device, dtype=torch.long)
                model_mean_guide = self.guide(model_mean, time_next_cond, model_kwargs=model_kwargs, ddim=True, train=False)
                model_mean = model_mean_guide

            x = model_mean + sigma * noise
            if time > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:]
        return x

    @torch.no_grad()
    def ddim_sample(self, shape, cond, **model_kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            50,
            1,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)
        cond = cond.to(device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                x, cond, time_cond, clip_x_start=self.clip_denoised, nn_cond=model_kwargs.get('nn_cond', None)
            )

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(x)

            model_mean = x_start * alpha_next.sqrt() + c * pred_noise

            x = model_mean + sigma * noise

        return x

    @torch.no_grad()
    def p_sample_loop(self, shape, x_start, x_cond, padding_mask=None):
        device = self.betas.device
        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), x_cond, padding_mask=padding_mask)    

        return x

    @torch.no_grad()
    def sample(self, x_start, x_cond, cond_mask=None, padding_mask=None):
        self.denoise_fn.eval()
        
        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask) + cond_mask * torch.randn_like(x_start).to(x_start.device)
            x_cond = torch.cat((x_cond, x_pose_cond), dim=-1)
       
        sample_res = self.p_sample_loop(x_start.shape, x_start, x_cond, padding_mask)
            
        self.denoise_fn.train()
        return sample_res

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, x_cond, t, noise=None, padding_mask=None):
        b, timesteps, d_input = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # The model_out should be the predicted noise
        model_out = self.denoise_fn(x, t, x_cond, padding_mask)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, reduction = 'none') * padding_mask[:, 0, 1:][:, :, None]
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none')

        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        
        return loss.mean()

    def forward(self, x_start, x_cond, cond_mask=None, padding_mask=None):
        bs = x_start.shape[0] 
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()
        
        # Only use hand poses as condition
        curr_loss = self.p_losses(x_start, x_cond, t, padding_mask=padding_mask)
        return curr_loss 