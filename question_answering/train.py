import argparse
import wandb
from model import Model
from dataset import Dataset
from trainer import Trainer


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base")
    parser.add_argument("--dataset_name", type=str, default="squad")
    parser.add_argument("--save_dir_path", type=str, default="./roberta_qa_ckpt")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser


if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()

    wandb.init('Question Answering')

    model = Model(model_name=args.model_name, save_dir_path=args.save_dir_path)

    train_dataset = Dataset(
        dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer_name,
        stage='train',
        batch_size=args.batch_size
    )

    valid_dataset = Dataset(
        dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer_name,
        stage='validation',
        batch_size=args.batch_size
    )

    train_dataloader = train_dataset.preproc_dataset()
    valid_dataloader = valid_dataset.preproc_dataset()

    trainer = Trainer(
        model=model,
        train_loader=train_dataloader,
        valid_loader=valid_dataloader
    )

    trainer.fit(max_epochs=3)