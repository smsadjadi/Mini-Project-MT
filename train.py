

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import time
import math
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################

from EDA import *
from dataloaders import *
from nets import *
from learning import *
from utils import *

input_lang, output_lang, pairs = prepareData('eng', 'spa')
MAX_LENGTH = max(len(pairs[-1][0]),len(pairs[-1][1]))
print('For Example: ',random.choice(pairs))

teacher_forcing_ratio = 0.5
SOS_token = 0
EOS_token = 1
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)

trainIters(encoder1, decoder1, pairs, n_iters=100000, print_every=5000, plot_every=100, learning_rate=0.01)

