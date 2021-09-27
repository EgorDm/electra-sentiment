from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
import re
import textwrap
import pandas as pd

import tensorflow.compat.v1 as tf
from tqdm import tqdm

import configure_finetuning
from finetune import preprocessing
from finetune import task_builder
from model import modeling, tokenization
from model import optimization
from run_finetuning import ModelRunner
from util import training_utils
from util import utils


class SentimentPrettyPrint:
    def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer, enabled=True) -> None:
        super().__init__()
        self.stream = open(os.path.join(config.data_dir, 'results', 'sentiment.out'), 'w') if print else None
        self.tokenizer = tokenizer
        self.enabled = enabled

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.stream.close()

    def write(self, text):
        if self.stream:
            self.stream.write(f'{text}\n')
        # print(text)

    def process(self, result):
        if not self.enabled: return

        features = [
            ['USDBTC-12h', 'USDBTC+12h', 'USDBTC+1d', 'USDBTC+2d', 'USDBTC+7d', 'USDBTC+14d'],
            ['USDETH-12h', 'USDETH+12h', 'USDETH+1d', 'USDETH+2d', 'USDETH+7d', 'USDETH+14d'],
            ['USDXRP-12h', 'USDXRP+12h', 'USDXRP+1d', 'USDXRP+2d', 'USDXRP+7d', 'USDXRP+14d']
        ]
        pattern = re.compile("|".join(map(re.escape, ['[PAD]', ' ##'])))
        text = pattern.sub('', ' '.join(self.tokenizer.convert_ids_to_tokens(result['input_ids']))).strip()
        targets = list(map(lambda x: f'{"+" if x >= 0 else ""}{x:.1f}', result['targets']))
        predictions = list(map(lambda x: f'{"+" if x >= 0 else ""}{x:.1f}', result['predictions']))
        self.write('')
        self.write(str(result['eid']).rjust(80, '='))
        self.write('\n'.join(textwrap.wrap(f'Input: {text}', 80)))
        for i, feature_set in enumerate(features):
            symbol = feature_set[0][:6]
            self.write(f'{symbol} ' + '-' * 73)
            cols = [f.replace(symbol, '').ljust(6) for f in feature_set]
            col_len = list(map(len, cols))
            self.write('type   |' + '|'.join(cols))
            self.write('target |' + '|'.join([v.ljust(l, ' ') for v, l in zip(targets[i*6:(i+1)*6], col_len)]))
            self.write('pred   |' + '|'.join([v.ljust(l, ' ') for v, l in zip(predictions[i*6:(i+1)*6], col_len)]))


def run_eval(config: configure_finetuning.FinetuningConfig):
    trial = 2
    heading_info = "model={:}".format(config.model_name)
    heading = lambda msg: utils.heading(msg + ": " + heading_info)
    heading("Config")
    utils.log_config(config)
    generic_model_dir = config.model_dir
    tasks = task_builder.get_tasks(config)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=config.vocab_file,
        do_lower_case=config.do_lower_case
    )

    split = "test2"
    config.model_dir = generic_model_dir + "_" + str(trial)
    config.do_train = False
    model_runner = ModelRunner(config, tasks)

    with SentimentPrettyPrint(config, tokenizer, False) as pretty:
        for task in tasks:
            predict_input_fn, _ = model_runner._preprocessor.prepare_predict(tasks, split)
            data = []
            results = model_runner._estimator.predict(input_fn=predict_input_fn, yield_single_examples=True)
            loader = tqdm(range(1_000_000))
            for i, r in enumerate(results):
                r = {k.replace(f'{task.name}_', ''): v for k, v in r.items()}
                # pretty.process(r)
                data.append({
                    'eid': r['eid'],
                    'output': r['predictions']
                })
                if i % 500 == 0:
                    loader.update(500)
                if i > 0 and i % 200_000 == 0:
                    df = pd.DataFrame(data)
                    df.to_parquet(os.path.join(config.data_dir, 'results2', f'outputs{i}.parquet'))

            df = pd.DataFrame(data)
            df.to_parquet(os.path.join(config.data_dir, 'results2', 'outputs.parquet'))

    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Location of data files (model weights, etc).")
    parser.add_argument("--model-name", required=True, help="The name of the model being fine-tuned.")
    parser.add_argument("--hparams", default="{}", help="JSON dict of model hyperparameters.")
    args = parser.parse_args()
    hparams = utils.load_json(args.hparams) if args.hparams.endswith(".json") else json.loads(args.hparams)
    tf.logging.set_verbosity(tf.logging.ERROR)
    run_eval(configure_finetuning.FinetuningConfig(args.model_name, args.data_dir, **hparams))


if __name__ == "__main__":
    main()
