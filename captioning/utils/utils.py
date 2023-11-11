from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
import copy
import math
import numpy as np
from .conAttModel import pack_wrapper, AttModel
from einops import rearrange, reduce
from captioning.models.gs_diffusion import GaussianDiffusion
from transformers.models.bert.modeling_bert import (BertEncoder,
                                                    BertPredictionHeadTransform,
                                                    BertLMPredictionHead,
                                                    BertEmbeddings)
from transformers.models.bert.configuration_bert import BertConfig