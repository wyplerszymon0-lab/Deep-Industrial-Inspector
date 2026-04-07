import torch.optim as optim

class DeepTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, images, labels):
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward() # Backpropagation - serce Deep Learningu
        self.optimizer.step()
        return loss.item()
