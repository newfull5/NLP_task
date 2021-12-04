from transformers import AutoModelForQuestionAnswering
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model_name):
        super(Model, self).__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def forward(self, batch):
        self.model(**batch)

    def save(self):
        pass

    def load(self):
        pass
