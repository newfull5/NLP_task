from transformers import AutoModelForQuestionAnswering
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, model_name, save_dir_path):
        super(Model, self).__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.safe_dir_path = save_dir_path

    def forward(self, batch):
        question_context, answer_span = batch
        outputs = self.model(**question_context)
        import pdb; pdb.set_trace()

    def save(self):
        self.model.save_pretrained(self.save_dir_path)

    def load(self):
        self.model.load_state_dict(
            torch.load(self.save_dir_path+'pytorch_model.bin', map_location=torch.device(self.device))
        )
