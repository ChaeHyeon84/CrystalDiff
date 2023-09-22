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

from DDPM import DDPM

def run(log_dir='logs/', train=True, **kwargs):
	
	config = _config()
	for key in kwargs.keys():
		assert key in config, 'wrong config arguments are given as an input.'

	config.update(kwargs)
	config['log_dir'] = log_dir
	config['train'] = train.lower()=='true'

	main(config)

@ex.automain
def main(_config):
    logger = pl.loggers.tensorboard.TensorBoardLogger(_config["log_dir"])
    transform=transforms.Compose([transforms.ToTensor(),  transforms.Resize((128, 128)),
                              transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])
    
    # Training
    if _config['train']:
        denoising_diffusion_model = DDPM(_config)
        train_data = DataLoader(mnist_train, batch_size=64)
        
        trainer = pl.Trainer(
                            accelerator=_config['accelerator'],
                            devices = _config['devices'],
                            num_nodes = _config['num_nodes'],
                            max_epochs = _config['max_epochs'],
                            precision = _config['precision'],
                            logger= logger,
                            # strategy= DDPStrategy(find_unused_paramters = True)
                            )
        trainer.fit(denoising_diffusion_model, train_data)
        
    # Sampling
    else:
        # denoising_diffusion_model.cuda()
        pass
    
    
        
    
    