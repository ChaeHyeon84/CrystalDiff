from pytorch_lightning.strategies.ddp import DDPStrategy
from denoising_diffusion_pytorch import Unet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase.io import read,write
from ase import Atoms, Atom
from ase.visualize import view
from collections import namedtuple

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import pymatgen as mg

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.optim as optim
from torchtext import data # torchtext.data 임포트
# from torchtext.data import Iterator
from torch.utils.data import Dataset, DataLoader, random_split
# from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

import csv
import pandas as pd

# from data import CIFData, AtomCustomJSONInitializer, GaussianDistance
import os
import csv
import random

from tqdm import tqdm
from config import config as _config
from config import ex

# beta_start = 0.0001
# beta_end = 0.02
# timesteps = 1500
# betas = torch.linspace(beta_start, beta_end, timesteps)
# betas = betas.type(torch.float64)
# timesteps = int(timesteps)
# alphas = 1-betas
# t = torch.randint(0, timesteps, (100, ), device='cpu')
# noise = torch.randn_like(z)

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(alphas, t, x_shape):
    b, *_ = t.shape
    out = alphas.gather(-1,t)
    return out.reshape(b, *((1,)) * (len(x_shape) -1))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale= 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

class GaussianDiffusion(nn.Module):
    def __init__(self, betas, model):
        super().__init__()
        
        betas = betas.type(torch.float64)
        timesteps = betas.shape[0]
        self.num_timesteps = int(timesteps)
        
        self.model =model
        self.self_condition = None
        sampling_timesteps = None
        
        alphas = 1. - betas        
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value = 1.)
        
        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        
        # helper function to register buffer from float64 to float32
        
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        
        register_buffer('sqrt_alphas_cumprod',  torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod)) 
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod-1))
        
        #calculations for posterior q(x_{t-1} | x_t, x_0)
        
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
        register_buffer('posterior_variance', posterior_variance)
        
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        sqrt_recip_alphas = torch.sqrt(1.0/alphas)
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        
        snr = alphas_cumprod / (1 - alphas_cumprod)
        
        min_snr_gamma = 5 
        maybe_clipped_snr = snr.clone()
        maybe_clipped_snr.clamp_(max = min_snr_gamma )
        
        register_buffer('loss_weight', maybe_clipped_snr / snr )
        
    @property
    def device(self):
        return self.betas.device
    
    def predict_start_from_noise(self, x_t, t, noise):
        return(
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t-
            extract(self.sqrt_recipm1_alphas_cumprod,t, x_t.shape) *noise
        )
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
        
    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        # if self.objective == 'pred_noise':
        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)

        if clip_x_start and rederive_pred_noise:
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        return imgs
    
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def training_losses(self, x_start, t, c=None, noise = None):
        b, c, h, w = x_start.shape
        noise= default(noise, lambda: torch.randn_like(x_start))
        
        # noise sample
        x = self.q_sample(x_start = x_start, t=t, noise =noise)
        x_self_cond = None
        model_out = self.model(x,t,x_self_cond)
        
        target = noise
        loss =F.mse_loss(model_out, target, reduction= 'none')
        loss = reduce(loss,'b ... -> b (...)', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
        