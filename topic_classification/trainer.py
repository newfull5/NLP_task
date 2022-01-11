from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch import optim
import torch.nn as nn
import torch
import wandb
 

class Trainer:
    def __init__(self, model, train_loader, valid_loader, val_check_step, lr):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.val_check_step = val_check_step
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)

    def _run_epoch(self, epoch):
        self.model.train()
        scaler = GradScaler()

        train_steps = 0
        total_train_loss = 0
        for batch in tqdm(self.train_loader, desc=f'train_epoch: {epoch}'):
            with autocast():
                outputs = self.model(batch)

            self.optimizer.zero_grad()
            scaler.scale(outputs.loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            train_steps += 1
            total_train_loss += float(outputs.loss)

        if train_steps % self.val_check_step == 0:
            wandb.log({'train_loss': total_train_loss / train_steps})
            self._valid_epoch(epoch)

    def _valid_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc='valid stage'):
                outputs = self.model(batch)
                total_val_loss += float(outputs.loss)
                val_steps += 1

        wandb.log({
            'val_loss': total_val_loss / val_steps,
            "epochs": epoch,
        })

    def fit(self, max_epoch):
        for epoch in range(max_epoch):
            self._run_epoch(epoch)
