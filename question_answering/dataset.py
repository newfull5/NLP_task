from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch
import multiprocessing as mp


class Dataset:
    def __init__(self, dataset_name, tokenizer_name, stage, batch_size):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.stage = stage
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.columns_to_return = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']

    def preproc_dataset(self):
        dataset = load_dataset(self.dataset_name)
        dataset = dataset[self.stage]
        dataset = dataset.map(self._convert_to_features, batched=True, batch_size=self.batch_size)
        dataset.set_format(type='torch', columns=self.columns_to_return)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=mp.cpu_count())

        return dataloader

    @staticmethod
    def _get_correct_alignement(context, answer):
        """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
        gold_text = answer['text'][0]
        start_idx = answer['answer_start'][0]
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx  # When the gold label position is good
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            return start_idx - 1, end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            return start_idx - 2, end_idx - 2  # When the gold label is off by two character
        else:
            raise ValueError()

    def _convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        input_pairs = list(zip(example_batch['context'], example_batch['question']))
        encodings = self.tokenizer.batch_encode_plus(input_pairs, pad_to_max_length=True, return_token_type_ids=True)

        # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
        start_positions, end_positions = [], []
        for i, (context, answer) in enumerate(zip(example_batch['context'], example_batch['answers'])):
            start_idx, end_idx = self._get_correct_alignement(context, answer)
            start_positions.append(encodings.char_to_token(i, start_idx))
            end_positions.append(encodings.char_to_token(i, end_idx - 1))

        encodings.update({'start_positions': start_positions,
                          'end_positions': end_positions})

        return encodings





