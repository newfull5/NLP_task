from model import Model
from dataset import QADataset

model = Model(model_name='roberta-base')
datasets = QADataset(dataset_name='squad_v2', tokenizer_name='roberta-base', stage='train')

