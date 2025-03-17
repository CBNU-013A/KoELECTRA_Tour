import random
import logging
import numpy as np
import torch

from seqeval import metrics as seqeval_metrics

from transformers import BertConfig, ElectraConfig
from transformers import ElectraTokenizer
from transformers import BertForTokenClassification, ElectraForTokenClassification

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

CONFIG_CLASSES = {
    "kobert": BertConfig,
    "koelectra-base": ElectraConfig,
    "koelectra-small": ElectraConfig,
    "koelectra-base-v2": ElectraConfig,
    "koelectra-base-v3": ElectraConfig,
    "koelectra-small-v2": ElectraConfig,
    "koelectra-small-v3": ElectraConfig,
}

TOKENIZER_CLASSES = {
    # "kobert": KoBertTokenizer,
    "koelectra-base": ElectraTokenizer,
    "koelectra-small": ElectraTokenizer,
    "koelectra-base-v2": ElectraTokenizer,
    "koelectra-base-v3": ElectraTokenizer,
    "koelectra-small-v2": ElectraTokenizer,
    "koelectra-small-v3": ElectraTokenizer,
}

MODEL_FOR_TOKEN_CLASSIFICATION = {
    "kobert": BertForTokenClassification,
    "koelectra-base": ElectraForTokenClassification,
    "koelectra-small": ElectraForTokenClassification,
    "koelectra-base-v2": ElectraForTokenClassification,
    "koelectra-base-v3": ElectraForTokenClassification,
    "koelectra-small-v2": ElectraForTokenClassification,
    "koelectra-small-v3": ElectraForTokenClassification,
    "koelectra-small-v3-51000": ElectraForTokenClassification,
}


def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }

def f1_pre_rec(labels, preds):
    return {
        "precision": seqeval_metrics.precision_score(labels, preds, suffix=True),
        "recall": seqeval_metrics.recall_score(labels, preds, suffix=True),
        "f1": seqeval_metrics.f1_score(labels, preds, suffix=True),
    }

def show_ner_report(labels, preds):
    return seqeval_metrics.classification_report(labels, preds, suffix=True)