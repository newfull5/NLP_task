import argparse
from model import Model
from dataset import NewsDataset
from tester import Tester
from torch.utils.data import DataLoader

args = argparse.Namespace(
    model_name="roberta-base",
    tokenizer_name="roberta-base",
    dataset_name="ag_news",
    save_dir_path="./roberta_tc_ckpt",
    device="cuda",
    batch_size=8,
    max_epochs=3,
    num_labels=4,
    patient=3,
    lr=3e-5
)

test_dataset = NewsDataset(
    tokenizer_name=args.tokenizer_name,
    dataset_name=args.dataset_name,
    stage='test'
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
)

tester = Tester(
    args=args,
    model=Model(model_name='dhtocks/Topic-Classification_temp', save_dir=''),
    test_loader=test_dataloader
)

tester.run()
tester.calc_score()