# NLP_task (wip)

## Topic Classification

- model: [roberta-base](https://huggingface.co/roberta-base)
- datasets: [ag_news](https://huggingface.co/datasets/ag_news)
- train_set: 120K -> train_set: 100K, valid_set: 20K
- test_Set: 7.6K
- label: World(0), Sports(1), Business(2), Sci/Tech(3)

```python3
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('dhtocks/Topic-Classification')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
```

## Named Entity Recognition

- model: [roberta-base](https://huggingface.co/roberta-base)
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

```python3
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('dhtocks/Topic-Classification')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
```