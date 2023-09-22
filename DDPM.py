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

import pytorch_lightning as pl
from diffusion import GaussianDiffusion, linear_beta_schedule

class DDPM(pl.LightningModule):
    def __init__(self, _config):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = _config
        self.timesteps = self.config['timesteps']
        self.channels = self.config['channels']
        self.optimizer = self.config['optimizer']
        self.lr = self.config['lr']
        
        self.model = Unet(
                    dim = self.config['dim'],
                    dim_mults = self.config['dim_mults'],
                    channels=self.config['channels'],
                    flash_attn = True
                )
        self.betas = linear_beta_schedule(timesteps= self.timesteps)
        
        self.diffusion = GaussianDiffusion(betas = self.betas, model= self.model)
        
        def forward(self, x):
            return self.diffusion.p_sample_loop(x.shape)
        
    def training_step(self, batch, batch_idx):
        x, _ = batch
        train_batch_size= x.shape[0]

        time = torch.randint(0, self.timesteps, (train_batch_size,)).long().to(x.device)
        loss = self.diffusion.training_losses(x, time)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):

        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError

        return optimizer

        
        
        