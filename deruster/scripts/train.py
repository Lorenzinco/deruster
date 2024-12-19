from deruster.model import DerusterModel
from deruster.dataset import DerusterDataset
from matplotlib import pyplot as plt

def train():
    train,validation = DerusterDataset.load("data/train")
    deruster_train = DerusterDataset(train)
    deruster_validation = DerusterDataset(validation, validation=True)
    model = DerusterModel(device="mps", dropout=0.1, num_layers=6, nhead=8, dim_model=8)
    train_performance, validation_performance = model.train(deruster_train, deruster_validation, epochs=100, lr=1e-4, batch_size=1)
    plt.plot(train_performance, label="Train")
    plt.plot(validation_performance, label="Validation")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()