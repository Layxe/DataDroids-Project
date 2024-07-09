from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

TRANSFORMERS_MAP = {
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased',
}

id2label   = {0: "entailment", 1: "neutral", 2: "contradiction"}
label2id   = {"entailment": 0, "neutral": 1, "contradiction": 2}
num_labels = len(id2label)

class TransformerHandler:

    def __init__(self, transformer_name, from_checkpoint=None):

        self.transformer_name = TRANSFORMERS_MAP[transformer_name]
        self.tokenizer        = AutoTokenizer.from_pretrained(self.transformer_name)

        if from_checkpoint is None:
            self.transformer      = AutoModelForSequenceClassification.from_pretrained(
                self.transformer_name,
                num_labels = num_labels,
                id2label   = id2label,
                label2id   = label2id
            )
        else:
            self.transformer = AutoModelForSequenceClassification.from_pretrained(from_checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id)

    def get_tokenizer(self):
        return self.tokenizer

    def get_transformer(self):
        return self.transformer

    def evaluation_function(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}