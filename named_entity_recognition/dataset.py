from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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
            tokenized = self.tokenizer(sample['token'], is_split_into_words=True, truncation=True)
            input_ids.append(tokenized['input_ids'])
            attention_mask.append(tokenized['attention_mask'])
            labels.append([-100 if i is None else sample['ner_tags'][i] for i in tokenized.word_ids()])

        return input_ids, attention_mask, labels

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

    def __len__(self):
        return len(self.input_ids)
