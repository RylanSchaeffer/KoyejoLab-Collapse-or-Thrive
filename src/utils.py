from datetime import datetime
import numpy as np
import random
import scipy
import scipy.special
import torch
import transformers
from typing import Dict


def format_time(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
