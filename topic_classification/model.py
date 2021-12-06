from transformers import AutoModelForSequenceClassification
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, model_name, save_dir):
        super(Model, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir

    def forward(self, batch):
        return self.model(**batch)

    def save(self):
        self.model.save_pretrained(self.save_dir)

    def load(self):
        self.model.load_state_dict(
            torch.load(self.save_dir+'pytorch_model.bin', map_location=torch.device(self.device))
        )