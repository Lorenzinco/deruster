import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random

VALIDATION_SPLIT = 0.2

class DerusterDataset(Dataset):

    def __init__(self, binaries, validation: bool = False):
        self.validation = validation
        self.binaries = binaries

    @staticmethod
    def load(path: str):
        assembly_path = os.path.join(path, "assembly")
        assembly = os.listdir(assembly_path)

        source_path = os.path.join(path, "source")
        source = os.listdir(source_path)

        binaries = []
        for asm_name in assembly:
            filename = asm_name.split(".")[0]
            source_name = filename + ".rs"
            if source_name not in source:
                continue
            binary = {"assembly" : os.path.join(assembly_path, asm_name), "source": os.path.join(source_path, source_name)}
            binaries.append(binary)
        
        random.shuffle(binaries)
        split = int(len(binaries) * VALIDATION_SPLIT)
        train = binaries[:split]
        validation = binaries[split:]
        return train, validation


    def __len__(self):
        length = 0
        length += len(self.binaries)
        return length
    
    def __getitem__(self, idx):
        binary = self.binaries[idx]
        with open(binary["assembly"], "r") as f:
            assembly = f.read()
        with open(binary["source"], "r") as f:
            source = f.read()
        return assembly, source
    
