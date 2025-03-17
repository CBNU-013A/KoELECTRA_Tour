import os
import copy
import json
import logging

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def convert_examples_to_features(
        args,
        examples,
        tokenizer,
        max_seq_length,
        pad_token_label_id=-100,
):
    label_lst = TourNerProcessor(args).get_labels() # get_labels() -> ["O", "B-ASP", "I-ASP", ...]
    label_map = {label: i for i, label in enumerate(label_lst)} # {"O": 0, "B-ASP": 1, "I-ASP": 2, ...}

    features = []

    # example단위 반복
    for (ex_index, example) in enumerate(examples):

        # 10000개 단위 로그 출력
        if ex_index % 10000 == 0:
            logger.info("Writing example {} of {}".format(ex_index, len(examples)))

        tokens = []
        label_ids = []

        # 문장의 단어들을 학습을 위해 토크나이징
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        special_tokens_count = 2
        # 최대 길이를 넘어가는 경우, max_seq_length - special_tokens_count 만큼 자름
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # Add [SEP]
        tokens += [tokenizer.sep_token]
        label_ids += [pad_token_label_id]

        # Add [CLS]
        tokens = [tokenizer.cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids

        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s " % " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_ids=label_ids)
        )
    return features

class TourNerProcessor(object):
    """Processor for the Tour NER data set """
    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["O", "B-ASP", "I-ASP", "B-OPI", "I-OPI", "B-LOC", "I-LOC", "B-PLC", "I-PLC"]

    @classmethod
    def _read_file(cls, input_file):
        """Read JSON file, and return a list of dictionaries (words & labels)"""
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(dataset):
            words = data['tokens']
            labels = data['bio_tagging']
            guid = "%s-%s" % (set_type, i)

            assert len(words) == len(labels)

            if i % 10000 == 0:
                logger.info(data)
            examples.append(InputExample(guid=guid, words=words, labels=labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir,
                                                        file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir,
                                                                  file_to_read)), mode)

def load_and_cache_examples(args, tokenizer, mode):
    # 1. NER Processor init
    processor = TourNerProcessor(args)
    # 2. Set the cached file path
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_len),
            mode
        )
        # e.g., cached_kobert_128_train
    )
    # 3. If the cached file exists, load it : examples
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file, weights_only=False)
    # 4. If the cached file does not exist, create it : examples
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise ValueError("For mode, only train, dev, test is avaiable")
        # 5. convert examples to features
        pad_token_label_id = CrossEntropyLoss().ignore_index # ignore padded token
        features = convert_examples_to_features(
            args,
            examples,
            tokenizer,
            max_seq_length=args.max_seq_len,
            pad_token_label_id=pad_token_label_id
        )
        # 6. Save the features into the cached file
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # 7. Convert features to tensor dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    # 8. Return the tensor dataset
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
    return dataset

processors = TourNerProcessor