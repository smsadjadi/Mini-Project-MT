

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

def prepareData(lang1, lang2):
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('
')
    pairs = [[normalizeString(s) for s in l.split('	')[:2]] for l in lines]
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    print("Read %s sentence pairs" %len(pairs))
    # pairs = filterPairs(pairs)
    # print("Considered %s sentence pairs" % len(pairs))
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

