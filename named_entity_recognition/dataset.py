from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch


# datasets = load_dataset('conll2003')
# train_set: 14K
# valid_set: 3.2K
# test_set: 3.4K
# label: {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

def collate_fn(batch):
    max_length = max([len(sample[0]) for sample in batch])
    input_ids_stack, attention_mask_stack, labels_stack = list(), list(), list()

    for input_ids, attention_mask, labels in batch:
        input_ids_stack.append(F.pad(torch.tensor(input_ids), pad=(0, max_length - len(input_ids))))
        attention_mask_stack.append(F.pad(torch.tensor(attention_mask), pad=(0, max_length - len(attention_mask))))
        labels_stack.append(F.pad(torch.tensor(labels), pad=(0, max_length - len(labels))))

    return torch.stack(input_ids_stack), torch.stack(attention_mask_stack), torch.stack(labels_stack)


class NerDataset(Dataset):
    def __init__(self, args, stage):
        super(NerDataset, self).__init__()
        self.args = args
        self.stage = stage
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        self.tokenizer.add_prefix_space = True
        self.input_ids, self.attention_mask, self.labels = self._preproc()

    def _preproc(self):
        datasets = load_dataset(self.args.dataset_name)
        datasets = datasets[self.stage]
        input_ids = []
        attention_mask = []
        labels = []

        for sample in datasets:
            tokenized = self.tokenizer(sample['tokens'], is_split_into_words=True, truncation=True)
            input_ids.append(tokenized['input_ids'])
            attention_mask.append(tokenized['attention_mask'])
            labels.append([-100 if i is None else sample['ner_tags'][i] for i in tokenized.word_ids()])

        return input_ids, attention_mask, labels

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

    def __len__(self):
        return len(self.input_ids)
