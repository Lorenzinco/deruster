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
            asm_functions_dir = os.path.join(assembly_path, filename)
            source_functions_dir = os.path.join(source_path, filename)
            if not os.path.exists(asm_functions_dir) or not os.path.exists(source_functions_dir):
                continue

            source_functions = os.listdir(source_functions_dir)
            if len(source_functions) == 0:
                continue
            for source_function in source_functions:
                asm_function = source_function.replace(".rs", ".s")
                if source_function not in os.listdir(source_functions_dir):
                    continue
                if source_function == "main.rs":
                    continue
                binary = {"assembly" : os.path.join(asm_functions_dir, asm_function), "source": os.path.join(source_functions_dir, source_function)}
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
    
