# NLP_task (wip)

## Topic Classification

- model: roberta-base
- datasets = [ag_news](https://huggingface.co/datasets/ag_news)
- train_set: 120K -> train_set: 100K, valid_set: 20K
- test_Set: 7.6K
- label: World(0), Sports(1), Business(2), Sci/Tech(3)

```python3
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('dhtocks/Topic-Classification')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
```