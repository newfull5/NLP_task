import argparse
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import AutoModelForTokenClassification, AutoTokenizer

class NerModel(pl.LightningModule):
    def __init__(self, args, model):
        super(NerModel, self).__init__()
        self.args = args
        self.model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids=input_ids.to(self.args.device),
            attention_mask=attention_mask.to(self.args.device),
            labels=labels.to(self.args.device)
        )
        return outputs.loss

    def training_step(self, input_ids, attention_mask, labels) -> STEP_OUTPUT:
        outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, input_ids, attention_mask, labels) -> Optional[STEP_OUTPUT]:
        outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", outputs.loss)
        return outputs.loss

    def test_step(self, input_ids, attention_mask, labels) -> Optional[STEP_OUTPUT]:
        outputs = self(input_ids, attention_mask, labels)
        self.log('test', outputs.logits, on_epoch=True)

    def configure_optimizers(self):
        pass



def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base")
    parser.add_argument("--dataset_name", type=str, default="conll2003")
    parser.add_argument("--save_dir_path", type=str, default="./roberta_tc_ckpt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_labels", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    return parser

