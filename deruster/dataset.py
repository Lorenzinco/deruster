import os
from deruster.binary import Binary
import torch
import numpy as np
from torch.utils.data import IterableDataset
import random
import tiktoken

VALIDATION_SPLIT = 0.2

class DerusterDataset(IterableDataset):

    def __init__(self, binaries, validation: bool = False):
        self.validation = validation
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self.binaries = binaries

    @staticmethod
    def load(path: str):
        assembly_path = os.path.join(path, "assembly")
        assembly = os.listdir(assembly_path)

        mir_path = os.path.join(path, "mir")
        mir = os.listdir(mir_path)

        binaries = []
        for asm_name in assembly:
            filename = asm_name.split(".")[0]
            mir_name = filename + ".mir"
            if mir_name not in mir:
                continue
            binary = Binary(os.path.join(assembly_path, asm_name), os.path.join(mir_path, mir_name))
            binaries.append(binary)
        
        random.shuffle(binaries)
        split = int(len(binaries) * VALIDATION_SPLIT)
        train = binaries[:split]
        validation = binaries[split:]
        return train, validation


    def __iter__(self):
        for binary in self.binaries:
            yield binary.binary, binary.correct

    def __len__(self):
        length = 0
        length += len(self.assembly)
        return length
    
    def __getitem__(self, idx):
        binary = self.binaries[idx]
        return binary.binary, binary.correct
