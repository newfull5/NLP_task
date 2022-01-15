import argparse
from typing import Optional
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS, EPOCH_OUTPUT
from transformers import AutoModelForTokenClassification, AutoTokenizer
from dataset import NerDataset, collate_fn
from pytorch_lightning.loggers import WandbLogger


class NerModel(pl.LightningModule):
    def __init__(self, args):
        super(NerModel, self).__init__()
        self.args = args
        self.model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        self.score = {}
        for class_num in range(self.args.num_labels):
            self.score[class_num] = {}
            for v in ['tp', 'tn', 'fp', 'fn']:
                self.score[class_num][v] = 0

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids=input_ids.to(self.args.device),
            attention_mask=attention_mask.to(self.args.device),
            labels=labels.to(self.args.device)
        )
        return outputs

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", outputs.loss)
        return outputs.loss

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        pred = torch.argmax(outputs.logits, dim=-1).squeeze()
        self._confusion_matrix(pred, labels)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        for class_num in range(self.args.num_labels):
            tp, tn, fp, fn = (self.score[class_num]['tp'],
                              self.score[class_num]['tn'],
                              self.score[class_num]['fp'],
                              self.score[class_num]['fn'])

            self.score[class_num]['precision'] = tp / (tp + fp)
            self.score[class_num]['recall'] = tp / (tp + fn)
            self.score[class_num]['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
            self.score[class_num]['f1_score'] = 2 * self.score[class_num]['precision'] * self.score[class_num]['recall'] / (
                    self.score[class_num]['precision'] + self.score[class_num]['recall'])

        print(self.score)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = NerDataset(args=self.args, stage='train')

        return DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        valid_dataset = NerDataset(args=self.args, stage='validation')

        return DataLoader(
            dataset=valid_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_dataset = NerDataset(args=self.args, stage='test')

        return DataLoader(
            dataset=test_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def _confusion_matrix(self, pred, labels):
        pred = pred.view(-1)
        labels = labels.view(-1)

        for class_num in range(self.args.num_labels):
            tp, tn, fp, fn = 0, 0, 0, 0
            for i in range(len(pred)):
                if labels[i] == -100:
                    continue

                # Positive
                if pred[i] == class_num:
                    if pred[i] == labels[i]:
                        tp += 1
                    if pred[i] != labels[i]:
                        fp += 1
                # Negative
                if pred[i] != class_num:
                    if labels[i] != class_num:
                        tn += 1
                    if labels[i] == class_num:
                        fn += 1

            self.score[class_num]['tp'] += tp
            self.score[class_num]['tn'] += tn
            self.score[class_num]['fp'] += fp
            self.score[class_num]['fn'] += fn


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base")
    parser.add_argument("--dataset_name", type=str, default="conll2003")
    parser.add_argument("--save_dir_path", type=str, default="./roberta_ner_ckpt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_labels", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    return parser


if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()

    pl.seed_everything(10_000)
    wandb_logger = WandbLogger(name="roberta_ner", project="NER")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="ckpt",
            filename="epoch={epoch}-valid_loss={valid_loss}",
            save_top_k=5,
            mode="min",
            auto_insert_metric_name=False,
        ),
        lr_monitor,
    ]

    trainer = pl.Trainer(
        val_check_interval=0.25,
        callbacks=callbacks,
        logger=wandb_logger,
        gpus=1,
        deterministic=True,
    )

    model = NerModel(args)
    trainer.fit(model)
    trainer.test()
