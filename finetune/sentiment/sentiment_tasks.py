import os

import numpy as np

import configure_finetuning
from finetune import task, feature_spec
import tensorflow.compat.v1 as tf
import pandas as pd

from finetune.classification import classification_metrics


class SentimentExample(task.Example):
    def __init__(self, eid, input_ids, targets):
        super(SentimentExample, self).__init__('sentiment')
        self.eid = eid
        self.input_ids = input_ids
        self.targets = targets


class SentimentTask(task.Task):
    def __init__(self, config: configure_finetuning.FinetuningConfig):
        super().__init__(config, "sentiment")
        self.n_outputs = 18
        self.pad_token_id = 1

    def get_prediction_module(self, bert_model, features, is_training, percent_done):
        reprs = bert_model.get_pooled_output()
        if is_training: reprs = tf.nn.dropout(reprs, keep_prob=0.9)

        predictions = tf.layers.dense(reprs, self.n_outputs)
        targets = features["targets"]
        losses = tf.keras.losses.mean_absolute_error(targets, predictions)
        outputs = dict(
            loss=losses,
            predictions=predictions,
            targets=targets,
            input_ids=features['input_ids'],
            eid=features["eid"]
        )
        return losses, outputs

    def get_scorer(self):
        return classification_metrics.RegressionScorer()

    def get_examples(self, split):
        table = pd.read_parquet(os.path.join('./sentiment_dataset', split))
        for i, row in table.iterrows():
            eid, input_ids, labels = row[['_id', 'input_ids', 'labels']]
            yield SentimentExample(eid, input_ids, labels)

    def featurize(self, example: SentimentExample, is_training, log=False):
        input_len = min(len(example.input_ids), self.config.max_seq_length)
        # Pad the input ids
        input_ids = np.full(shape=self.config.max_seq_length, fill_value=self.pad_token_id)
        input_ids[:input_len] = example.input_ids[:input_len]
        # Create a attaion mask
        input_mask = np.zeros((self.config.max_seq_length,), dtype=np.int)
        input_mask[:input_len] = 1

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "targets": example.targets,
            "eid": example.eid,
            "task_id": self.config.task_names.index(self.name),
        }

    def get_feature_specs(self):
        return [
            feature_spec.FeatureSpec("eid", []),
            feature_spec.FeatureSpec("targets", [self.n_outputs], is_int_feature=False),
        ]
