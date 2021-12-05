from torch.cuda.amp import autocast, GradScaler
from torch import optim
import torch
import wandb


class Trainer:
    def __init__(self, model, train_loader, valid_loader):
        super(Trainer, self).__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=3e-5)

    def _train_epoch(self, epoch):
        self.model.train()
        scaler = GradScaler()

        for batch in self.train_loader:
            batch = batch.to(self.device)

            with autocast():
                outputs = self.model(batch)

            self.optimizer.zero_grad()
            scaler.scale(outputs.loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            wandb.log(
                {
                    'loss': outputs.loss
                }
            )

    def _valid_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for batch in self.valid_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                wandb.log({'val_loss': outputs.loss})

    def fit(self, max_epochs):
        for epoch in range(max_epochs):
            self._train_epoch(epoch)
            self._valid_epoch(epoch)
