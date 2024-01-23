

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

import os
import yaml
import torch
import numpy as np
import torch.nn as nn
from skimage import color
import matplotlib.pyplot as plt

from EDA import *
from dataloaders import *
from nets import *
from learning import *
from losses import *
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
encoder1.load_state_dict(torch.load("./saved/encoder.pth")) ; encoder1.train()
decoder1.load_state_dict(torch.load("./saved/decoder.pth")) ; decoder1.train()

evaluateRandomly(encoder1, attn_decoder1)

