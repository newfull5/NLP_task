from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch import nn

# train_set: 120K -> train_set: 100K, valid_set: 20K
# test_set: 7.6K
# label: World(0), Sports(1), Business(2), Sci/Tech(3)
tokenizer = AutoTokenizer.from_pretrained('roberta_base')

datasets = load_dataset('ag_news')

valid_text = datasets['train']['text'][:20000]
valid_label = datasets['train']['label'][:20000]

train_text = datasets['train']['text'][20000:]
train_label = datasets['train']['label'][20000:]


class NewsDataset(Dataset):
    def __init__(self, tokenizer_name, dataset_name, stage):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.text, self.label = self._split_dataset(dataset_name, stage)

    def _split_dataset(self, dataset_name, stage):
        datasets = load_dataset(dataset_name)

        if stage == 'test':
            text = datasets['test']['text']
            label = datasets['test']['label']

        elif stage == 'valid':
            text = datasets['valid']['text'][:20000]
            label = datasets['valid']['label'][:20000]

        elif stage == 'train':
            text = datasets['train']['text'][20000:]
            label = datasets['train']['label'][20000:]

        else:
            raise Exception("you can set stage only 'train', 'test' or 'valid'")

        text = [self._preproc_text(t) for t in text]

        return text, label

    def _preproc_text(self, text):
        tokenized = self.tokenizer(
            text=text,
            padding=True,
            truncation=True, return_tensors='pt'
        )
        return tokenized

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.text[idx], self.label[idx]