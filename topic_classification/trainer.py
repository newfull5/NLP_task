from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch import optim
import torch.nn as nn
import torch
import wandb


class Trainer:
    def __init__(self, model, train_loader, valid_loader, val_check_interval, lr):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.val_check_interval = val_check_interval
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)

    def _run_epoch(self, epoch):
        self.model.train()
        scaler = GradScaler()

        for batch in tqdm(self.train_loader, desc=f'train_epoch: {epoch}'):
            with autocast():
                outputs = self.model(batch)

            self.optimizer.zero_grad()
            scaler.scale(outputs.loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            wandb.log({'loss': outputs.loss})

    def _valid_stage(self):
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc='valid stage'):
                outputs = self.model(batch)
                wandb.log({'val_loss': outputs.loss})

    def fit(self, max_epoch, patient):
        for epoch in range(max_epoch):
            self._run_epoch(epoch)