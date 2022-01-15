# Topic Classificaion

## Abstract

- model: roberta-base
- datasets: [ag_news](https://huggingface.co/datasets/ag_news)
- train_set: 120K -> train_set: 100K, valid_set: 20K
- test_Set: 7.6K
- label: World(0), Sports(1), Business(2), Sci/Tech(3)


## Usage

```python3
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('dhtocks/Topic-Classification')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
```

- fine tuning 
```shell
python3 train.py --model_name MODEL_NAME \
                 --tokenizer_name TOKENIZER_NAME \
                 --batch_size = BATCH_SIZE \
                 --save_dir_path = SAVE_DIR_PATH 
```

## Score

|               | precision | recall | f1_score | Accuracy       |
|---------------|-----------|--------|----------|----------------|
| World(0)      | 0.9449    | 0.9489 | 0.9469   |                |
| Sports(1)     | 0.9873    | 0.9826 | 0.9849   |                |
| Business(2)   | 0.9028    | 0.89   | 0.9208   |                |
| Sci/Tech(3)   | 0.9023    | 0.9284 | 0.9151   |                |
| Macro Average | 0.9376    | 0.9375 | 0.9374   | 0.9375 (Total) |


## Config

- Model: roberta-base
- datasets: ag_news
- train : valid : test = 100K, 20K, 7.6K
- batch_size = 8
- device: P100 x 1 (Colab)
- lr = 3e-5 (no warmup, no scheduling)
- mixed_precision