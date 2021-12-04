from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_dataset
import torch


class QADataset(Dataset):
    def __init__(self, dataset_name, tokenizer_name, stage):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.stage = stage
        self.dataset_name = dataset_name
        self.questions_contexts = []
        self.answers_span = []
        self._preproc_dataset()

    def _preproc_dataset(self):
        dataset = load_dataset(self.dataset_name)
        dataset = dataset[self.stage]

        for i in range(len(dataset)):
            context = dataset[i]['context']
            question = dataset[i]['question']

            question_context = self.tokenizer(
                context,
                question,
                padding='max_length',
                truncation='only_first',
                return_tensors='pt'
            )

            answer = dataset[i]['answers']['text'][0]
            answer_start = dataset[i]['answers']['answer_start'][0]

            answer_token_start = len(self.tokenizer(context[:answer_start])['input_ids'])-2
            answer_token_end = answer_token_start + len(self.tokenizer(answer)['input_ids'])-2

            answer_span = torch.tensor([answer_token_start, answer_token_end])

            self.questions_contexts.append(question_context)
            self.answers_span.append(answer_span)

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return self.questions_contexts[idx], self.answers[idx]

QADataset(dataset_name='squad_v2', stage='train')
