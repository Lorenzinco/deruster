import torch
import torch.nn as nn
from tqdm.auto import tqdm
from deruster.dataset import DerusterDataset
import transformers

class DerusterEncoder(nn.Module):
    def __init__(self, device: str = "cpu", dropout: float = 0.1, num_layers: int = 6, nhead: int = 8, dim_model: int = 512):
        super().__init__()
        self.computation_device = device
        self.dropout = dropout
        self.asm_tokenizer = transformers.AutoTokenizer.from_pretrained('LLM4Binary/llm4decompile-22b-v2')
        self.code_tokenizer = transformers.AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.asm_tokenizer.pad_token = self.asm_tokenizer.eos_token
        self.code_tokenizer.pad_token = self.code_tokenizer.eos_token
        self.embedding = nn.Embedding(self.asm_tokenizer.vocab_size, dim_model,device=self.computation_device)

        self.convolution = nn.Sequential(
            nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nhead,batch_first=True, dropout=dropout,
            dim_feedforward=4*dim_model, activation='gelu',device=self.computation_device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,norm=nn.LayerNorm(dim_model),
            enable_nested_tensor=False)

    def forward(self, x: list[str]):
        x = self.asm_tokenizer(x, padding=True, return_tensors='pt')
        unembedded_tokens   = x['input_ids'].to(self.computation_device)
        tokens = self.embedding(unembedded_tokens)
        tokens = tokens.permute(0,2,1)
        tokens = self.convolution(tokens)
        tokens = tokens.permute(0,2,1)
        attention_mask = x['attention_mask']
        attention_mask = attention_mask.bool().to(self.computation_device)
        x = self.transformer(tokens, src_key_padding_mask=attention_mask)
        return x



class DerusterDecoder(nn.Module):
    def __init__(self, device: str = "cpu", dropout: float = 0.1, num_layers: int = 6, nhead: int = 8, dim_model: int = 512):
        super().__init__()
        self.computation_device = device
        self.dropout = dropout

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=nhead, batch_first=True, dropout=dropout,
            dim_feedforward=4*dim_model, activation='gelu',device=self.computation_device)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        x = self.transformer(x)
        return x
    
class DerusterModel(nn.Module):
    def __init__(self, device: str = "cpu", dropout: float = 0.1, num_layers: int = 6, nhead: int = 8, dim_model: int = 32):
        super().__init__()
        self.computation_device = device
        self.dropout = dropout
        self.encoder = DerusterEncoder(device=device, dropout=dropout, num_layers=num_layers, nhead=nhead, dim_model=dim_model)
        self.decoder = DerusterDecoder(device=device, dropout=dropout, num_layers=num_layers, nhead=nhead, dim_model=dim_model)

    def forward(self, assembly: list[str], current_generated_code: list[str]):
        x = self.encoder(assembly)
        #x = self.decoder(assembly)
        return x

    def train(self, train: DerusterDataset, validation: DerusterDataset, epochs: int = 10, lr: float = 1e-4, batch_size: int = 8):
        self.to(self.computation_device)
        train_performance = []
        validation_performance = []
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            batches = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            avg_loss = 0
            for bn, (binary, correct) in enumerate(batches):
                optimizer.zero_grad()
                #binary = binary.to(self.computation_device)
                #correct = correct.to(self.computation_device)
                output = self.forward(binary, correct)
                loss = criterion(output, correct)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                avg_loss += loss.item()
                batches.set_postfix(loss=avg_loss / (bn + 1))
            train_performance.append(avg_loss)

            performance = self.evaluate(validation=validation, batch_size=batch_size)
            validation_performance.append(performance)
        return train_performance, validation_performance
    
    def evaluate(self, validation: DerusterDataset, batch_size: int = 32):
        self.to(self.computation_device)
        performance = []
        train_loader = torch.utils.data.DataLoader(dataset=validation, batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            batches = tqdm(train_loader, desc=f"Validation")
            avg_loss = 0
            for bn, (binary, correct) in enumerate(batches):
                binary = binary.to(self.computation_device)
                correct = correct.to(self.computation_device)
                output = self.forward(binary)
                loss = criterion(output, correct)
                loss.backward()
                avg_loss += loss.item()
                batches.set_postfix(loss=avg_loss / (bn + 1))
            performance.append(avg_loss)
        return performance
    
    def _preprocess_asm(self, assembly: str):
        return self.asm_tokenizer(assembly, padding=True, return_tensors='pt')
    
    def _preprocess_source(self, source: str):
        return self.source_tokenizer(source, padding=True, return_tensors='pt')