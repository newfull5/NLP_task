from transformers import AutoModelForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
import torch


class Model(nn.Module):
    def __init__(self, model_name, save_dir):
        super(Model, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir

    def forward(self, batch):
        inputs, labels = batch
        inputs['input_ids'] = self._move_to_cuda(inputs['input_ids'].squeeze())
        inputs['attention_mask'] = self._move_to_cuda(inputs['attention_mask'])
        labels = self._move_to_cuda(labels)

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )

        return outputs

    def _move_to_cuda(self, inputs):
        if torch.is_tensor(inputs):
            return inputs.to(self.device)
        elif isinstance(inputs, list):
            return [self._move_to_cuda(x) for x in inputs]
        elif isinstance(inputs, dict):
            return {key: self._move_to_cuda(value) for key, value in inputs.items()}
        else:
            return inputs

    def save(self):
        self.model.save_pretrained(self.save_dir)

    def load(self):
        self.model.load_state_dict(
            torch.load(self.save_dir+'pytorch_model.bin', map_location=torch.device(self.device))
        )