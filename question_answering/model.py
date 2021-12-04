from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model_name, num_label):
        super(Model, self).__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def forward(self, batch):
        self.model(**batch)

    def save(self):
        pass

    def load(self):
        pass
