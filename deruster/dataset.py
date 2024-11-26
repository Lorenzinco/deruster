import os
import torch
import numpy as np
from torch.utils.data import IterableDataset
import random

class DerusterDataset(IterableDataset):
    def __init__(self,assembly: list[list[torch.Tensor]], validation: bool = False):
        self.assembly = assembly
        self.validation = validation
    
    def __len__(self):
        length = 0
        length += len(self.assembly)
        return length
    
    
    

