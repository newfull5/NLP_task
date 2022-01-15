# Named Entity Recognition

## Abstract

- model: roberta-base
- datasets: [conll2003_noMISC](https://huggingface.co/datasets/Davlan/conll2003_noMISC)
- train_set: 14K
- valid_set: 3.2K
- test_set: 3.4K
- label: 
  - O (0)
  - B-PER (1)
  - I-PER (2)
  - B-ORG (3)
  - I-ORG (4)
  - B-LOC (5)
  - I-LOC (6)



## Usage

```python3
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('dhtocks/Named-Entity-Recognition')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
```

- fine tuning 
```shell
python3 train.py --model_name MODEL_NAME \
                 --tokenizer_name TOKENIZER_NAME \
                 --batch_size = BATCH_SIZE 
```

## Score

|               | precision | recall | f1_score | Accuracy |
|---------------|-----------|--------|----------|----------|
| O (0)         | 0.9985    | 0.9958 | 0.9971   | 0.9948   |
| B-PER (1)     | 0.9517    | 0.9385 | 0.9451   | 0.9979   |
| I-PER (2)     | 0.9796    | 0.9783 | 0.9789   | 0.9993   |
| B-ORG (3)     | 0.8650    | 0.9097 | 0.8868   | 0.9953   |
| I-ORG (4)     | 0.8206    | 0.9124 | 0.8641   | 0.9976   |
| B-LOC (5)     | 0.8918    | 0.9239 | 0.9075   | 0.9966   |
| I-LOC (6)     | 0.8015    | 0.8589 | 0.8292   | 0.9991   |
| Macro Average | 0.9012    | 0.9310 | 0.9245   | 0.9972   |


## Config

- Model: roberta-base
- datasets: conll2003_noMISC
- train : valid : test = 14, 3.2, 3.7K
- batch_size = 16
- device: P100 x 1 (Colab)
- lr = 3e-5 (no warmup, no scheduling)
