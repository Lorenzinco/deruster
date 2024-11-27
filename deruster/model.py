import torch
import torch.nn as nn
from tqdm.auto import tqdm
from deruster.dataset import DerusterDataset

class DerusterEncoder(nn.Module):
    def __init__(self, device: str = "cpu", dropout: float = 0.1, num_layers: int = 6, nhead: int = 8, dim_model: int = 512):
        super().__init__()
        self.computation_device = device
        self.dropout = dropout

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor):
        x = self.transformer(x)
        return x



class DerusterDecoder(nn.Module):
    def __init__(self, device: str = "cpu", dropout: float = 0.1, num_layers: int = 6, nhead: int = 8, dim_model: int = 512):
        super().__init__()
        self.computation_device = device
        self.dropout = dropout

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=nhead)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor):
        x = self.transformer(x)
        return x
    
class DerusterModel(nn.Module):
    def __init__(self, device: str = "cpu", dropout: float = 0.1, num_layers: int = 6, nhead: int = 8, dim_model: int = 512):
        super().__init__()
        self.computation_device = device
        self.dropout = dropout
        self.encoder = DerusterEncoder(device=device, dropout=dropout, num_layers=num_layers, nhead=nhead, dim_model=dim_model)
        self.decoder = DerusterDecoder(device=device, dropout=dropout, num_layers=num_layers, nhead=nhead, dim_model=dim_model)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train(self, train: DerusterDataset, validation: DerusterDataset, epochs: int = 10, lr: float = 1e-3, batch_size: int = 32):
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
                binary = binary.to(self.computation_device)
                correct = correct.to(self.computation_device)
                output = self.forward(binary)
                loss = criterion(output, correct)
                loss.backward()
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