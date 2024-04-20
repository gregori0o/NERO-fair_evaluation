import graph_tool as gt

import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import argparse
from enum import Enum
import json
import random
import torch
import time

import nero.constants as constants
import nero.converters.tudataset as tudataset
import nero.converters.ogbdataset as ogbdataset
import nero.converters.iamdataset as iamdataset
import nero.embedding.pipelines as pipelines
import nero.tools.logging as logging

logger = logging.get_configured_logger()

R_EVALUATION = 3
SEED = 12
experiment_name = time.strftime("%Y_%m_%d_%Hh%Mm%Ss")


class DatasetName(Enum):
    DD = "DD"
    NCI1 = "NCI1"
    PROTEINS = "PROTEINS_full"
    ENZYMES = "ENZYMES"
    IMDB_BINARY = "IMDB-BINARY"
    IMDB_MULTI = "IMDB-MULTI"
    REDDIT_BINARY = "REDDIT-BINARY"
    REDDIT_MULTI = "REDDIT-MULTI-5K"
    COLLAB = "COLLAB"
    MUTAG = "MUTAG"
    MOLHIV = "ogbg-molhiv"
    # iam datasets
    WEB = "Web"


IAM_DATASETS = [DatasetName.WEB]


def load_indexes(dataset_name: DatasetName):
    path = f"data_splits/{dataset_name.value}.json"
    with open(path, "r") as f:
        indexes = json.load(f)
    return indexes


def evaluate(proba, predictions, targets):
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average="micro")
    recall = recall_score(targets, predictions, average="micro")
    f1 = f1_score(targets, predictions, average="micro")
    macro_f1 = f1_score(targets, predictions, average="macro")
    if proba.shape[1] == 2:
        proba = proba[:, 1]
    roc = roc_auc_score(targets, proba, average="macro", multi_class="ovr")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro f1": macro_f1,
        "roc": roc,
    }


def perform_experiment(dataset: DatasetName) -> None:
    dataset_name = dataset.value
    print(f"Running experiment {experiment_name} on {dataset_name}")
    if dataset == DatasetName.MOLHIV:
        samples, classes, description = ogbdataset.ogbdataset2persisted(dataset_name)
    elif dataset in IAM_DATASETS:
        samples, classes, description = iamdataset.iamdataset2persisted(dataset_name)
    else:
        samples, classes, description = tudataset.tudataset2persisted(dataset_name)
    pipeline = pipelines.create_pipeline(description, 'AV0', (20, 20, None))

    csv_dir_path = pathlib.Path(constants.CSV_DIR)
    csv_dir_path.mkdir(parents=True, exist_ok=True)
    csv_file = f"{logging.formatted_today()}.csv"
    with open(csv_dir_path / csv_file, 'a') as file:
        file.write("Benchmark,Repetition,Split,Accuracy,Macro F1,ROC")

    indexes = load_indexes(dataset)
    for i, fold in enumerate(indexes):
        print(f"FOLD {i}")
        test_idx = fold["test"]
        train_idx = fold["train"]
        # test_idx = list(range(2))
        # train_idx = list(range(2, 10))

        train_samples = [samples[i] for i in train_idx]
        train_classes = [classes[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]
        test_classes = [classes[i] for i in test_idx]

        for j in range(R_EVALUATION):

            # setting seed
            seed = SEED + j
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            pipeline.fit(train_samples, train_classes)
            proba = pipeline.predict_proba(test_samples)
            prediction = pipeline.predict(test_samples)
            metrics = evaluate(proba, prediction, test_classes)
            # accuracy_score = pipeline.score(test_samples, test_classes)
            with open(csv_dir_path / csv_file, 'a') as file:
                file.write(f"\n{dataset_name},{j},{i},{metrics['accuracy']},{metrics['macro f1']},{metrics['roc']}")

    df = pd.read_csv(csv_dir_path / csv_file)
    df = df.groupby(
        ['Benchmark', 'Split'], as_index=False,
    ).aggregate(
        {
            'Accuracy': [np.mean],
            'Macro F1': [np.mean],
            'ROC': [np.mean],
        }
    ).groupby(
        ['Benchmark'], as_index=False,
    ).aggregate(
        {
            ('Accuracy', 'mean'): [np.mean, np.std],
            ('Macro F1', 'mean'): [np.mean, np.std],
            ('ROC', 'mean'): [np.mean, np.std],
        }
    )
    print(df)
    df.to_csv(f"results/{experiment_name}_{dataset_name}.csv")


if __name__ == "__main__":
    # # test local
    # perform_experiment(DatasetName.MUTAG)
    # exit()
    parser = argparse.ArgumentParser("run_experiment")
    parser.add_argument("--dataset", help="Dataset name", type=str, choices=[d.value for d in DatasetName] + ["all"], default="all")
    args = parser.parse_args()
    if args.dataset == "all":
        for dataset_name in DatasetName:
            print(f"Running experiment on {dataset_name}")
            perform_experiment(dataset_name)
    else:
        dataset_name = DatasetName(args.dataset)
        print(f"Running experiment on {dataset_name}")
        perform_experiment(dataset_name)