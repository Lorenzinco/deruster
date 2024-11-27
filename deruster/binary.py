import torch
import tiktoken


class Binary():
    def __init__(self, binary_path: str, correct_path: str):
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        with open(binary_path, "rb") as binary_file:
            binary = binary_file.read().decode("utf-8", errors="ignore")
            self.binary = torch.tensor(tokenizer.encode(binary))
        with open(correct_path, "rb") as correct_file:
            correct = correct_file.read().decode("utf-8", errors="ignore")
            self.correct = torch.tensor(tokenizer.encode(correct))
