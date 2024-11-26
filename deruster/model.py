import torch
import torch.nn as nn
from tqdm.auto import tqdm

class Deruster(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.computation_device = device