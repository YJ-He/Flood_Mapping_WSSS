import os
import torch
import random
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_seed(seed=0):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_sum(self):
        return self.sum

