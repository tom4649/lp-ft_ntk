# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Custom metric for PubMed experiments."""

import datasets
from sklearn.metrics import f1_score


def simple_accuracy(preds, labels):
    return float((preds == labels).mean())


def acc_and_f1(preds, labels, f1_avg="binary"):
    acc = simple_accuracy(preds, labels)
    f1 = float(f1_score(y_true=labels, y_pred=preds, average=f1_avg))
    return {
        "accuracy": acc,
        "f1": f1,
    }


def evaluate_multirc(ids_preds, labels):
    """
    Computes F1 score and Exact Match for MultiRC predictions.
    """
    question_map = {}
    for id_pred, label in zip(ids_preds, labels):
        question_id = "{}-{}".format(
            id_pred["idx"]["paragraph"], id_pred["idx"]["question"]
        )
        pred = id_pred["prediction"]
        if question_id in question_map:
            question_map[question_id].append((pred, label))
        else:
            question_map[question_id] = [(pred, label)]
    f1s, ems = [], []
    for question, preds_labels in question_map.items():
        question_preds, question_labels = zip(*preds_labels)
        f1 = f1_score(y_true=question_labels, y_pred=question_preds, average="macro")
        f1s.append(f1)
        em = int(sum([p == l for p, l in preds_labels]) == len(preds_labels))
        ems.append(em)
    f1_m = float((sum(f1s) / len(f1s)))
    em = sum(ems) / len(ems)
    f1_a = float(
        f1_score(
            y_true=labels,
            y_pred=[id_pred["prediction"] for id_pred in ids_preds],
        )
    )
    return {"exact_match": em, "f1_m": f1_m, "f1_a": f1_a}


class PubMed(datasets.Metric):
    def _info(self):
        if self.config_name not in ["pubmed"]:
            raise KeyError(
                "You should supply a configuration name selected in "
                '["amazon", "dynasent", "semeval", "sst5"]'
            )
        return datasets.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(self._get_feature_types()),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _get_feature_types(self):
        return {
            "predictions": datasets.Value("int64"),
            "references": datasets.Value("int64"),
        }

    def _compute(self, predictions, references):
        return {"accuracy": simple_accuracy(predictions, references)}
