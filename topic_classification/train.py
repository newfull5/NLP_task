import argparse
import wandb
from model import Model
import multiprocessing as mp
from torch.utils.data import DataLoader
from dataset import NewsDataset
from trainer import Trainer


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base")
    parser.add_argument("--dataset_name", type=str, default="ag_news")
    parser.add_argument("--save_dir_path", type=str, default="./roberta_tc_ckpt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--patient", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    return parser


if __name__ =='__main__':
    parser = _get_parser()
    args = parser.parse_args()

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
        val_check_interval=1,
        lr=args.lr
    )

    trainer.fit(args.max_epochs, patient=args.patient)