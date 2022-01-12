import torch
from tqdm import tqdm


class Tester:
    def __init__(self, args, model, test_loader):
        self.model = model.to(args.device).eval()
        self.args = args
        self.labels = []
        self.pred = []
        self.test_loader = test_loader
        self.score = {}

    def run(self):
        for batch in tqdm(self.test_loader):
            logits, _ = self.model(batch)
            pred = torch.argmax(logits, dim=-1).tolist()

            _, labels = batch
            self.pred += pred
            self.labels += labels.tolist()

    def calc_score(self):
        self._confusion_matrix()
        self._f1_score()
        self._accuracy()
        self._macro_average()
        print(self.score)

    def _confusion_matrix(self):
        for class_num in range(self.args.num_labels):
            tp, tn, fp, fn = 0, 0, 0, 0
            for i in range(len(self.pred)):
                # Positive
                if self.pred[i] == class_num:
                    if self.pred[i] == self.labels[i]:
                        tp += 1
                    if self.pred[i] != self.labels[i]:
                        fp += 1
                # Negative
                if self.pred[i] != class_num:
                    if self.labels[i] != class_num:
                        tn += 1
                    if self.labels[i] == class_num:
                        fn += 1

            self.score[class_num] = {}
            self.score[class_num]['confusion_matrix'] = (tp, tn, fp, fn)

    def _f1_score(self):
        for class_num in range(self.args.num_labels):
            tp, tn, fp, fn = self.score[class_num]['confusion_matrix']

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * precision * recall / (precision + recall)

            self.score[class_num]['precision'] = precision
            self.score[class_num]['recall'] = recall
            self.score[class_num]['f1_score'] = f1_score

    def _macro_average(self):
        self.score['macro_average'] = {}
        for metric in ['precision', 'recall', 'f1_score']:
            self.score['macro_average'][metric] = sum(
                [self.score[class_num][metric] for class_num in range(self.args.num_labels)]
            ) / self.args.num_labels

    def _accuracy(self):
        cnt = 0
        for i in range(len(self.pred)):
            if self.pred[i] == self.labels[i]:
                cnt += 1

        self.score['accuracy'] = cnt / len(self.pred)
