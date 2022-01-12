import argparse
import wandb
from model import Model
from dataset import NewsDataset
from trainer import Trainer
from torch.utils.data import DataLoader

def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base")
    parser.add_argument("--dataset_name", type=str, default="ag_news")
    parser.add_argument("--save_dir_path", type=str, default="./roberta_tc_ckpt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--patient", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    return parser


args = argparse.Namespace(
  model_name="roberta-base",
  tokenizer_name="roberta-base",
  dataset_name="ag_news",
  save_dir_path="./roberta_tc_ckpt",
  batch_size=8,
  max_epochs=3,
  patient=3,
  lr=3e-5
)

wandb.init('Topic Classification')

model = Model(model_name=args.model_name, save_dir=args.save_dir_path)

train_dataset = NewsDataset(
    tokenizer_name=args.tokenizer_name,
    dataset_name=args.dataset_name,
    stage='train'
)

valid_dataset = NewsDataset(
    tokenizer_name=args.tokenizer_name,
    dataset_name=args.dataset_name,
    stage='valid'
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
)

valid_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=args.batch_size,
    shuffle=True,
)

trainer = Trainer(
    model=model,
    train_loader=train_dataloader,
    valid_loader=valid_dataloader,
    val_check_step=2000,
    lr=args.lr
)

trainer.fit(args.max_epochs)
