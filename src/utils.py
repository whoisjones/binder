import os
import logging
import collections
import json
from math import log
import copy
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True)
class Annotation:
    id: str
    entity_type: str
    start_char: int
    end_char: int
    text: str

@dataclass
class Span:
    type_id: int
    start: int
    end: int
    start_mask: List[int]
    end_mask: List[int]
    span_mask: List[int]

def compute_tp_fn_fp(predictions: Set, labels: Set, **kwargs) -> Dict[str, float]:
    # tp, fn, fp
    if len(predictions) == 0:
        return {"tp": 0, "fn": len(labels), "fp": 0}
    if len(labels) == 0:
        return {"tp": 0, "fn": 0, "fp": len(predictions)}
    tp = len(predictions & labels)
    fn = len(labels) - tp
    fp = len(predictions) - tp
    return {"tp": tp, "fn": fn, "fp": fp}


def compute_precision_recall_f1(tp: int, fn: int, fp: int, **kwargs) -> Dict[str, float]:
    if tp + fp + fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if tp + fp == 0:
        return {"precision": 1.0, "recall": .0, "f1": .0}
    if tp + fn == 0:
        return {"precision": .0, "recall": 1.0, "f1": .0}
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1}


def postprocess_nested_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
    id_to_type: List[str],
    max_span_length: int = 30,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
    tokenizer = None,
    **kwargs,
) -> Dict:
    logger.setLevel(log_level)

    if len(predictions) != 3:
        raise ValueError("`predictions` should be a tuple with three elements (start_logits, end_logits, span_logits).")
    all_start_logits, all_end_logits, all_span_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The gold annotations.
    all_annotations = set()
    # The dictionaries we have to fill.
    all_predictions = set()

    entity_type_vocab = list(set(id_to_type))
    entity_type_count = collections.defaultdict(int)
    metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}
    start_metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}
    end_metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}

    # Logging.
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        example_annotations = set()
        example_predictions = set()

        # Looping through all NER annotations.
        for entity_type, start_char, end_char in zip(
            example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
            entity_type_count["all"] += 1
            entity_type_count[entity_type] += 1
            example_annotations.add(Annotation(
                id=example["id"],
                entity_type=entity_type,
                start_char=start_char,
                end_char=end_char,
                text=example["text"][start_char:end_char]
            ))

        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the masks for start and end indices.
            token_start_mask = np.array(features[feature_index]["token_start_mask"]).astype(bool)
            token_end_mask = np.array(features[feature_index]["token_end_mask"]).astype(bool)

            # We grab the predictions of the model for this feature.
            span_logits = all_span_logits[feature_index]
            # We use the [CLS] logits as thresholds
            span_preds = np.triu(span_logits > span_logits[:, 0:1, 0:1])

            type_ids, start_indexes, end_indexes = (
                token_start_mask[np.newaxis, :, np.newaxis] & token_end_mask[np.newaxis, np.newaxis, :] & span_preds
            ).nonzero()

            # This is what will allow us to map some the positions in our logits to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all start and end indices.
            for type_id, start_index, end_index in zip(type_ids, start_indexes, end_indexes):
                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # Don't consider spans with a length that is > max_span_length.
                if end_index - start_index + 1 > max_span_length:
                    continue
                # A prediction contains (example_id, entity_type, start_index, end_index)
                start_char, end_char = offset_mapping[start_index][0], offset_mapping[end_index][1]
                pred = Annotation(
                    id=example["id"],
                    entity_type=id_to_type[type_id],
                    start_char=start_char,
                    end_char=end_char,
                    text=example["text"][start_char:end_char],
                )
                example_predictions.add(pred)

        for t in metrics_by_type.keys():
            for k, v in compute_tp_fn_fp(
                example_predictions if t == "all" else set(filter(lambda x: x.entity_type == t, example_predictions)),
                example_annotations if t == "all" else set(filter(lambda x: x.entity_type == t, example_annotations)),
            ).items():
                metrics_by_type[t][k] += v

        all_annotations.update(example_annotations)
        all_predictions.update(example_predictions)

        example_gold_starts = set((x.entity_type, x.start_char) for x in example_annotations)
        example_pred_starts = set((x.entity_type, x.start_char) for x in example_predictions)
        for t in start_metrics_by_type.keys():
            for k, v in compute_tp_fn_fp(
                example_pred_starts if t == "all" else set(filter(lambda x: x[0] == t, example_pred_starts)),
                example_gold_starts if t == "all" else set(filter(lambda x: x[0] == t, example_gold_starts)),
            ).items():
                start_metrics_by_type[t][k] += v

        example_gold_ends = set((x.entity_type, x.end_char) for x in example_annotations)
        example_pred_ends = set((x.entity_type, x.end_char) for x in example_predictions)
        for t in end_metrics_by_type.keys():
            for k, v in compute_tp_fn_fp(
                example_pred_ends if t == "all" else set(filter(lambda x: x[0] == t, example_pred_ends)),
                example_gold_ends if t == "all" else set(filter(lambda x: x[0] == t, example_gold_ends)),
            ).items():
                end_metrics_by_type[t][k] += v

    metrics = collections.OrderedDict()
    precisions, recalls = [], []
    sorted_entity_types = ["all"] + sorted(entity_type_vocab, key=lambda x: entity_type_count[x], reverse=True)
    for x, x_metrics_by_type in {"span": metrics_by_type, "start": start_metrics_by_type, "end": end_metrics_by_type}.items():
        metrics[x] = {}
        for t in sorted_entity_types:
            metrics_for_t = compute_precision_recall_f1(**x_metrics_by_type[t])
            f1, precision, recall = metrics_for_t["f1"], metrics_for_t["precision"], metrics_for_t["recall"]
            if x == "span" and t != "all":
                precisions.append(precision)
                recalls.append(recall)
            metrics[x][t] = {}
            for k, v in metrics_for_t.items():
                metrics[x][t][k] = v

    for t in sorted_entity_types:
        support = entity_type_count[t]
        logger.info(f"***** {t} ({support}) *****")
        for x in metrics:
            f1, precision, recall = metrics[x][t]["f1"], metrics[x][t]["precision"], metrics[x][t]["recall"]
            logger.info(f"F1 = {f1:>6.1%}, Precision = {precision:>6.1%}, Recall = {recall:>6.1%} (for {x})")

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        # Convert flat predictions to hierarchical.
        example_id_to_predictions = {}
        for pred in all_predictions:
            example_id = pred.id
            if example_id not in example_id_to_predictions:
                example_id_to_predictions[example_id] = set()
            example_id_to_predictions[example_id].add((pred.start_char, pred.end_char, pred.entity_type, pred.text))

        predictions_to_save = []
        for example in examples:
            example = copy.deepcopy(example)
            example.pop("word_start_chars")
            example.pop("word_end_chars")
            gold_ner = set()
            for entity_type, start_char, end_char in zip(
                example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
                gold_ner.add((start_char, end_char, entity_type, example["text"][start_char:end_char]))
            example.pop("entity_types")
            example.pop("entity_start_chars")
            example.pop("entity_end_chars")

            pred_ner = example_id_to_predictions.get(example["id"], set())
            example["gold_ner"] = sorted(gold_ner)
            example["pred_ner"] = sorted(pred_ner)
            predictions_to_save.append(example)

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            for pred in predictions_to_save:
                writer.write(json.dumps(pred) + "\n")

        metric_file = os.path.join(
            output_dir, "metrics.json" if prefix is None else f"{prefix}_metrics.json"
        )

        logger.info(f"Saving metrics to {metric_file}.")
        with open(metric_file, "a") as writer:
            writer.write(json.dumps(metrics) + "\n")

    metrics = {
        "f1": metrics["span"]["all"]["f1"],
        "precision": metrics["span"]["all"]["precision"],
        "recall": metrics["span"]["all"]["recall"]
    }

    return {
        "predictions": all_predictions,
        "labels": all_annotations,
        "metrics": metrics,
    }


def search_insert(array: List, x) -> int:
    l, r = 0, len(array) - 1
    while l <= r:
        m = (l + r) // 2
        if array[m] == x:
            return m
        elif array[m] < x:
            l = m + 1
        else:
            r = m - 1
    return l


def remove_overlaps(pred_scores: Dict[Annotation, float]) -> Set:
    predictions, starts = [], []
    for pred in sorted(pred_scores, key=lambda x: (-pred_scores[x], x.start_char, x.end_char)):
        start, end = pred.start_char, pred.end_char
        if len(predictions) == 0:
            predictions.append(pred)
            starts.append(start)
        else:
            index = search_insert(starts, start)
            if index == 0:
                next_start = predictions[index].start_char
                if end <= next_start:
                    predictions.insert(index, pred)
                    starts.insert(index, start)
            elif index == len(predictions):
                prev_end = predictions[index - 1].end_char
                if start >= prev_end:
                    predictions.insert(index, pred)
                    starts.insert(index, start)
            else:
                next_start = predictions[index].start_char
                prev_end = predictions[index - 1].end_char
                if start >= prev_end and end <= next_start:
                    predictions.insert(index, pred)
                    starts.insert(index, start)
    return set(predictions)


def convert_to_iob(
    text: str,
    word_start_chars: List[int],
    word_end_chars: List[int],
    entity_start_chars: List[int],
    entity_end_chars: List[int],
    entity_types: List[str],
    **kwargs
) -> Dict[str, List[str]]:
    words = [text[s:e] for s, e in zip(word_start_chars, word_end_chars)]
    labels = []
    pos = 0
    while pos < len(word_start_chars):
        start, end = word_start_chars[pos], word_end_chars[pos]
        if start in entity_start_chars:
            index = entity_start_chars.index(start)
            labels.append("B-" + entity_types[index])
            if end in entity_end_chars:
                assert index == entity_end_chars.index(end), breakpoint()
            else:
                while end not in entity_end_chars:
                    pos += 1
                    start, end = word_start_chars[pos], word_end_chars[pos]
                    assert start not in entity_start_chars, breakpoint()
                    labels.append("I-" + entity_types[index])
                assert index == entity_end_chars.index(end), breakpoint()
        else:
            labels.append("O")
        pos += 1
    assert len(words) == len(labels), breakpoint()
    return {"words": words, "labels": labels}


def error_analysis(annotations: Set[Annotation], predictions: Set[Annotation], entity_types: List[str]) -> Dict:
    fp = [p for p in predictions if p not in annotations]
    fn = [a for a in annotations if a not in predictions]
    fp_counter = collections.Counter([(x.text, x.entity_type) for x in fp])
    fn_counter = collections.Counter([(x.text, x.entity_type) for x in fn])
    fp_patterns = '|'.join(sorted(set([k[0].lower() for k, _ in fp_counter.most_common(30)])))
    ret = collections.OrderedDict({
        "most-common fp patterns": fp_patterns,
        "most-common fp errors": {" | ".join(k): v for k, v in fp_counter.most_common(30)},
        "most-common fn errors": {" | ".join(k): v for k, v in fn_counter.most_common(30)},
    })
    for t in entity_types:
        fp_counter = collections.Counter([x.text for x in fp if x.entity_type == t])
        fn_counter = collections.Counter([x.text for x in fn if x.entity_type == t])
        fp_patterns = '|'.join(sorted(set([k.lower() for k, _ in fp_counter.most_common(10)])))
        ret.update({
            t: {
                "most-common fp patterns": fp_patterns,
                "most-common fp errors": {k: v for k, v in fp_counter.most_common(10)},
                "most-common fn errors": {k: v for k, v in fn_counter.most_common(10)},
            }
        })
    return ret

def postprocess_flat_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
    id_to_type: List[str],
    max_span_length: int = 30,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
    tokenizer = None,
    **kwargs,
) -> Dict:
    logger.setLevel(log_level)

    if len(predictions) != 3:
        raise ValueError("`predictions` should be a tuple with three elements (start_logits, end_logits, span_logits).")
    all_start_logits, all_end_logits, all_span_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The gold annotations.
    all_annotations = set()
    # The dictionaries we have to fill.
    all_predictions = set()

    entity_type_vocab = list(set(id_to_type))
    entity_type_count = collections.defaultdict(int)
    metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}
    start_metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}
    end_metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}

    # Logging.
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        example_annotations = set()
        example_pred_scores = {}

        # Looping through all NER annotations.
        for entity_type, start_char, end_char in zip(
            example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
            entity_type_count["all"] += 1
            entity_type_count[entity_type] += 1
            example_annotations.add(Annotation(
                id=example["id"],
                entity_type=entity_type,
                start_char=start_char,
                end_char=end_char,
                text=example["text"][start_char:end_char]
            ))

        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the masks for start and end indices.
            token_start_mask = np.array(features[feature_index]["token_start_mask"]).astype(bool)
            token_end_mask = np.array(features[feature_index]["token_end_mask"]).astype(bool)

            # We grab the predictions of the model for this feature.
            span_logits = all_span_logits[feature_index]
            # We use the [CLS] logits as thresholds
            span_preds = np.triu(span_logits > span_logits[:, 0:1, 0:1])

            type_ids, start_indexes, end_indexes = (
                token_start_mask[np.newaxis, :, np.newaxis] & token_end_mask[np.newaxis, np.newaxis, :] & span_preds
            ).nonzero()

            # This is what will allow us to map some the positions in our logits to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all start and end indices.
            for type_id, start_index, end_index in zip(type_ids, start_indexes, end_indexes):
                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # Don't consider spans with a length that is > max_span_length.
                if end_index - start_index + 1 > max_span_length:
                    continue
                # A prediction contains (example_id, entity_type, start_index, end_index)
                start_char, end_char = offset_mapping[start_index][0], offset_mapping[end_index][1]
                pred = Annotation(
                    id=example["id"],
                    entity_type=id_to_type[type_id],
                    start_char=start_char,
                    end_char=end_char,
                    text=example["text"][start_char:end_char],
                )
                if pred in example_pred_scores:
                    example_pred_scores[pred] = max(example_pred_scores[pred], span_logits[type_id][start_index][end_index])
                else:
                    example_pred_scores[pred] = span_logits[type_id][start_index][end_index]

        example_predictions = remove_overlaps(example_pred_scores)

        for t in metrics_by_type.keys():
            for k, v in compute_tp_fn_fp(
                example_predictions if t == "all" else set(filter(lambda x: x.entity_type == t, example_predictions)),
                example_annotations if t == "all" else set(filter(lambda x: x.entity_type == t, example_annotations)),
            ).items():
                metrics_by_type[t][k] += v

        all_annotations.update(example_annotations)
        all_predictions.update(example_predictions)

        example_gold_starts = set((x.entity_type, x.start_char) for x in example_annotations)
        example_pred_starts = set((x.entity_type, x.start_char) for x in example_predictions)
        for t in start_metrics_by_type.keys():
            for k, v in compute_tp_fn_fp(
                example_pred_starts if t == "all" else set(filter(lambda x: x[0] == t, example_pred_starts)),
                example_gold_starts if t == "all" else set(filter(lambda x: x[0] == t, example_gold_starts)),
            ).items():
                start_metrics_by_type[t][k] += v

        example_gold_ends = set((x.entity_type, x.end_char) for x in example_annotations)
        example_pred_ends = set((x.entity_type, x.end_char) for x in example_predictions)
        for t in end_metrics_by_type.keys():
            for k, v in compute_tp_fn_fp(
                example_pred_ends if t == "all" else set(filter(lambda x: x[0] == t, example_pred_ends)),
                example_gold_ends if t == "all" else set(filter(lambda x: x[0] == t, example_gold_ends)),
            ).items():
                end_metrics_by_type[t][k] += v

    metrics = collections.OrderedDict()
    precisions, recalls = [], []
    sorted_entity_types = ["all"] + sorted(entity_type_vocab, key=lambda x: entity_type_count[x], reverse=True)
    for x, x_metrics_by_type in {"span": metrics_by_type, "start": start_metrics_by_type, "end": end_metrics_by_type}.items():
        metrics[x] = {}
        for t in sorted_entity_types:
            metrics_for_t = compute_precision_recall_f1(**x_metrics_by_type[t])
            f1, precision, recall = metrics_for_t["f1"], metrics_for_t["precision"], metrics_for_t["recall"]
            if x == "span" and t != "all":
                precisions.append(precision)
                recalls.append(recall)
            metrics[x][t] = {}
            for k, v in metrics_for_t.items():
                metrics[x][t][k] = v

    for t in sorted_entity_types:
        support = entity_type_count[t]
        logger.info(f"***** {t} ({support}) *****")
        for x in metrics:
            f1, precision, recall = metrics[x][t]["f1"], metrics[x][t]["precision"], metrics[x][t]["recall"]
            logger.info(f"F1 = {f1:>6.1%}, Precision = {precision:>6.1%}, Recall = {recall:>6.1%} (for {x})")

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        # Convert flat predictions to hierarchical.
        example_id_to_predictions = {}
        for pred in all_predictions:
            example_id = pred.id
            if example_id not in example_id_to_predictions:
                example_id_to_predictions[example_id] = set()
            example_id_to_predictions[example_id].add((pred.start_char, pred.end_char, pred.entity_type, pred.text))

        predictions_to_save = []
        for example in examples:
            example = copy.deepcopy(example)
            example.pop("word_start_chars")
            example.pop("word_end_chars")
            gold_ner = set()
            for entity_type, start_char, end_char in zip(
                example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
                gold_ner.add((start_char, end_char, entity_type, example["text"][start_char:end_char]))
            example.pop("entity_types")
            example.pop("entity_start_chars")
            example.pop("entity_end_chars")

            pred_ner = example_id_to_predictions.get(example["id"], set())
            example["gold_ner"] = sorted(gold_ner)
            example["pred_ner"] = sorted(pred_ner)

            predictions_to_save.append(example)

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            for pred in predictions_to_save:
                writer.write(json.dumps(pred) + "\n")

        metric_file = os.path.join(
            output_dir, "metrics.json" if prefix is None else f"{prefix}_metrics.json"
        )

        logger.info(f"Saving metrics to {metric_file}.")
        with open(metric_file, "a") as writer:
            writer.write(json.dumps(metrics) + "\n")

    metrics = {
        "f1": metrics["span"]["all"]["f1"],
        "precision": metrics["span"]["all"]["precision"],
        "recall": metrics["span"]["all"]["recall"]
    }

    return {
        "predictions": all_predictions,
        "labels": all_annotations,
        "metrics": metrics,
    }

def postprocess_flat_bce_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
    id_to_type: List[str],
    max_span_length: int = 30,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
    tokenizer = None,
    **kwargs,
) -> Dict:
    logger.setLevel(log_level)

    if len(predictions) != 3:
        raise ValueError("`predictions` should be a tuple with three elements (start_logits, end_logits, span_logits).")
    all_start_logits, all_end_logits, all_span_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The gold annotations.
    all_annotations = set()
    # The dictionaries we have to fill.
    all_predictions = set()

    entity_type_vocab = list(set(id_to_type))
    entity_type_count = collections.defaultdict(int)
    metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}
    start_metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}
    end_metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}

    # Logging.
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        example_annotations = set()
        example_pred_scores = {}

        # Looping through all NER annotations.
        for entity_type, start_char, end_char in zip(
            example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
            entity_type_count["all"] += 1
            entity_type_count[entity_type] += 1
            example_annotations.add(Annotation(
                id=example["id"],
                entity_type=entity_type,
                start_char=start_char,
                end_char=end_char,
                text=example["text"][start_char:end_char]
            ))

        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the masks for start and end indices.
            token_start_mask = np.array(features[feature_index]["token_start_mask"]).astype(bool)
            token_end_mask = np.array(features[feature_index]["token_end_mask"]).astype(bool)

            span_logits = all_span_logits[feature_index]
            span_probs = 1.0 / (1.0 + np.exp(-span_logits))
            # We use the [CLS] logits as thresholds
            span_preds = np.triu(span_probs > 0.8)

            type_ids, start_indexes, end_indexes = (
                token_start_mask[np.newaxis, :, np.newaxis] & token_end_mask[np.newaxis, np.newaxis, :] & span_preds
            ).nonzero()

            # This is what will allow us to map some the positions in our logits to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all start and end indices.
            for type_id, start_index, end_index in zip(type_ids, start_indexes, end_indexes):
                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # Don't consider spans with a length that is > max_span_length.
                if end_index - start_index + 1 > max_span_length:
                    continue
                # A prediction contains (example_id, entity_type, start_index, end_index)
                start_char, end_char = offset_mapping[start_index][0], offset_mapping[end_index][1]
                pred = Annotation(
                    id=example["id"],
                    entity_type=id_to_type[type_id],
                    start_char=start_char,
                    end_char=end_char,
                    text=example["text"][start_char:end_char],
                )
                if pred in example_pred_scores:
                    example_pred_scores[pred] = max(example_pred_scores[pred], span_logits[type_id][start_index][end_index])
                else:
                    example_pred_scores[pred] = span_logits[type_id][start_index][end_index]

        example_predictions = remove_overlaps(example_pred_scores)

        for t in metrics_by_type.keys():
            for k, v in compute_tp_fn_fp(
                example_predictions if t == "all" else set(filter(lambda x: x.entity_type == t, example_predictions)),
                example_annotations if t == "all" else set(filter(lambda x: x.entity_type == t, example_annotations)),
            ).items():
                metrics_by_type[t][k] += v

        all_annotations.update(example_annotations)
        all_predictions.update(example_predictions)

        example_gold_starts = set((x.entity_type, x.start_char) for x in example_annotations)
        example_pred_starts = set((x.entity_type, x.start_char) for x in example_predictions)
        for t in start_metrics_by_type.keys():
            for k, v in compute_tp_fn_fp(
                example_pred_starts if t == "all" else set(filter(lambda x: x[0] == t, example_pred_starts)),
                example_gold_starts if t == "all" else set(filter(lambda x: x[0] == t, example_gold_starts)),
            ).items():
                start_metrics_by_type[t][k] += v

        example_gold_ends = set((x.entity_type, x.end_char) for x in example_annotations)
        example_pred_ends = set((x.entity_type, x.end_char) for x in example_predictions)
        for t in end_metrics_by_type.keys():
            for k, v in compute_tp_fn_fp(
                example_pred_ends if t == "all" else set(filter(lambda x: x[0] == t, example_pred_ends)),
                example_gold_ends if t == "all" else set(filter(lambda x: x[0] == t, example_gold_ends)),
            ).items():
                end_metrics_by_type[t][k] += v

    metrics = collections.OrderedDict()
    precisions, recalls = [], []
    sorted_entity_types = ["all"] + sorted(entity_type_vocab, key=lambda x: entity_type_count[x], reverse=True)
    for x, x_metrics_by_type in {"span": metrics_by_type, "start": start_metrics_by_type, "end": end_metrics_by_type}.items():
        metrics[x] = {}
        for t in sorted_entity_types:
            metrics_for_t = compute_precision_recall_f1(**x_metrics_by_type[t])
            f1, precision, recall = metrics_for_t["f1"], metrics_for_t["precision"], metrics_for_t["recall"]
            if x == "span" and t != "all":
                precisions.append(precision)
                recalls.append(recall)
            metrics[x][t] = {}
            for k, v in metrics_for_t.items():
                metrics[x][t][k] = v

    for t in sorted_entity_types:
        support = entity_type_count[t]
        logger.info(f"***** {t} ({support}) *****")
        for x in metrics:
            f1, precision, recall = metrics[x][t]["f1"], metrics[x][t]["precision"], metrics[x][t]["recall"]
            logger.info(f"F1 = {f1:>6.1%}, Precision = {precision:>6.1%}, Recall = {recall:>6.1%} (for {x})")

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        # Convert flat predictions to hierarchical.
        example_id_to_predictions = {}
        for pred in all_predictions:
            example_id = pred.id
            if example_id not in example_id_to_predictions:
                example_id_to_predictions[example_id] = set()
            example_id_to_predictions[example_id].add((pred.start_char, pred.end_char, pred.entity_type, pred.text))

        predictions_to_save = []
        for example in examples:
            example = copy.deepcopy(example)
            example.pop("word_start_chars")
            example.pop("word_end_chars")
            gold_ner = set()
            for entity_type, start_char, end_char in zip(
                example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
                gold_ner.add((start_char, end_char, entity_type, example["text"][start_char:end_char]))
            example.pop("entity_types")
            example.pop("entity_start_chars")
            example.pop("entity_end_chars")

            pred_ner = example_id_to_predictions.get(example["id"], set())
            example["gold_ner"] = sorted(gold_ner)
            example["pred_ner"] = sorted(pred_ner)

            predictions_to_save.append(example)

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            for pred in predictions_to_save:
                writer.write(json.dumps(pred) + "\n")

        metric_file = os.path.join(
            output_dir, "metrics.json" if prefix is None else f"{prefix}_metrics.json"
        )

        logger.info(f"Saving metrics to {metric_file}.")
        with open(metric_file, "a") as writer:
            writer.write(json.dumps(metrics) + "\n")

    metrics = {
        "f1": metrics["span"]["all"]["f1"],
        "precision": metrics["span"]["all"]["precision"],
        "recall": metrics["span"]["all"]["recall"]
    }

    return {
        "predictions": all_predictions,
        "labels": all_annotations,
        "metrics": metrics,
    }

# def postprocess_flat_bce_predictions(
#     examples,
#     features,
#     predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
#     id_to_type: List[str],
#     max_span_length: int = 30,
#     output_dir: Optional[str] = None,
#     prefix: Optional[str] = None,
#     log_level: Optional[int] = logging.WARNING,
#     tokenizer = None,
#     **kwargs,
# ) -> Dict:
#     logger.setLevel(log_level)

#     if len(predictions) != 3:
#         raise ValueError("`predictions` should be a tuple with three elements (start_logits, end_logits, span_logits).")
#     all_start_logits, all_end_logits, all_span_logits = predictions

#     if len(predictions[0]) != len(features):
#         raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

#     # ---- decoding kwargs (sane defaults) ----
#     temp              = float(kwargs.get("temp", 1.0))              # temperature scaling for logits
#     start_threshold   = float(kwargs.get("start_threshold", 0.7))   # gate on start prob
#     end_threshold     = float(kwargs.get("end_threshold", 0.7))     # gate on end prob
#     span_threshold    = float(kwargs.get("span_threshold", 0.8))    # gate on fused prob; float or {type: thr}
#     alpha             = float(kwargs.get("alpha", 0.7))             # fusion weight for start logit
#     beta              = float(kwargs.get("beta", 0.7))              # fusion weight for end logit
#     localmax_window   = int(kwargs.get("localmax_window", 1))       # 0 disables local-max gating
#     span_top_k        = kwargs.get("span_top_k", None)              # int or None (per-type top-K after gating)

#     # Build a map example -> its features.
#     example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
#     features_per_example = collections.defaultdict(list)
#     for i, feature in enumerate(features):
#         features_per_example[example_id_to_index[feature["example_id"]]].append(i)

#     # The gold annotations / predictions.
#     all_annotations = set()
#     all_predictions = set()

#     entity_type_vocab = list(set(id_to_type))
#     entity_type_count = collections.defaultdict(int)
#     metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}
#     start_metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}
#     end_metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}

#     logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

#     def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

#     def _localmax_1d(prob_2d: np.ndarray, w: int) -> np.ndarray:
#         """prob_2d: [T,L] -> bool peaks [T,L] within ±w."""
#         if w <= 0:
#             return np.ones_like(prob_2d, dtype=bool)
#         try:
#             from numpy.lib.stride_tricks import sliding_window_view
#             pad = np.pad(prob_2d, ((0, 0), (w, w)), mode="edge")          # [T, L+2w]
#             win = sliding_window_view(pad, window_shape=(2*w+1), axis=1)   # [T, L, 2w+1]
#             return prob_2d == win.max(axis=-1)
#         except Exception:
#             T, L = prob_2d.shape
#             out = np.ones((T, L), dtype=bool)
#             for d in range(1, w+1):
#                 out &= (prob_2d >= np.roll(prob_2d, +d, axis=1))
#                 out &= (prob_2d >= np.roll(prob_2d, -d, axis=1))
#             return out

#     # Loop over all examples
#     for example_index, example in enumerate(examples):
#         example_annotations = set()
#         example_pred_scores = {}

#         # gold
#         for entity_type, start_char, end_char in zip(
#             example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
#             entity_type_count["all"] += 1
#             entity_type_count[entity_type] += 1
#             example_annotations.add(Annotation(
#                 id=example["id"],
#                 entity_type=entity_type,
#                 start_char=start_char,
#                 end_char=end_char,
#                 text=example["text"][start_char:end_char]
#             ))

#         # features for this example
#         for feature_index in features_per_example[example_index]:
#             token_start_mask = np.asarray(features[feature_index]["token_start_mask"], dtype=bool)
#             token_end_mask   = np.asarray(features[feature_index]["token_end_mask"],   dtype=bool)
#             default_span_mask = np.asarray(features[feature_index]["default_span_mask"], dtype=bool)
#             offset_mapping = features[feature_index]["offset_mapping"]

#             # --- logits (float32) + temperature ---
#             start_logits = all_start_logits[feature_index].astype(np.float32) / temp   # [T,L]
#             end_logits   = all_end_logits[feature_index].astype(np.float32)   / temp   # [T,L]
#             span_logits  = all_span_logits[feature_index].astype(np.float32)  / temp   # [T,L,L]

#             # --- probabilities for gating (keep logits for ranking) ---
#             start_prob = _sigmoid(start_logits)
#             end_prob   = _sigmoid(end_logits)
#             span_prob  = _sigmoid(span_logits)

#             T = span_prob.shape[0]
#             L = span_prob.shape[-1]

#             # --- validity / length masks ---
#             i_idx = np.arange(L)[:, None]
#             j_idx = np.arange(L)[None, :]
#             length_mask = ((j_idx - i_idx + 1) <= max_span_length)[None, :, :]

#             valid_mask = (
#                 token_start_mask[None, :, None] &
#                 token_end_mask[None, None, :] &
#                 default_span_mask[None, :, :] &
#                 length_mask
#             )

#             # --- local maximum gating on boundaries ---
#             start_is_peak = _localmax_1d(start_prob, localmax_window)   # [T,L]
#             end_is_peak   = _localmax_1d(end_prob,   localmax_window)   # [T,L]

#             # --- fused score in logit space (better ranking) ---
#             joint_logit = span_logits + alpha * start_logits[:, :, None] + beta * end_logits[:, None, :]
#             joint_prob  = _sigmoid(joint_logit)

#             # --- thresholds (span_threshold supports scalar or dict per type) ---
#             if isinstance(span_threshold, dict):
#                 thr = np.array([span_threshold.get(id_to_type[t], 0.8) for t in range(T)], dtype=np.float32)[:, None, None]
#             else:
#                 thr = float(span_threshold)

#             gate = (
#                 (start_prob >= start_threshold)[:, :, None] &
#                 (end_prob   >= end_threshold)[:, None, :] &
#                 start_is_peak[:, :, None] &
#                 end_is_peak[:, None, :] &
#                 (joint_prob >= thr)
#             )

#             keep = gate & valid_mask

#             # --- optional per-type top-K pruning AFTER gating ---
#             if isinstance(span_top_k, int) and span_top_k > 0:
#                 keep_flat  = keep.reshape(T, -1)
#                 score_flat = joint_logit.reshape(T, -1)
#                 new_keep = np.zeros_like(keep_flat, dtype=bool)
#                 for t in range(T):
#                     idx = np.where(keep_flat[t])[0]
#                     if idx.size == 0: continue
#                     k = min(span_top_k, idx.size)
#                     sel = idx[np.argpartition(-score_flat[t, idx], kth=k-1)[:k]]
#                     new_keep[t, sel] = True
#                 keep = new_keep.reshape(keep.shape)

#             type_ids, start_indexes, end_indexes = keep.nonzero()

#             # map to char spans
#             for type_id, start_index, end_index in zip(type_ids, start_indexes, end_indexes):
#                 if (
#                     start_index >= len(offset_mapping)
#                     or end_index >= len(offset_mapping)
#                     or offset_mapping[start_index] is None
#                     or offset_mapping[end_index] is None
#                 ):
#                     continue
#                 if end_index - start_index + 1 > max_span_length:
#                     continue
#                 start_char, end_char = offset_mapping[start_index][0], offset_mapping[end_index][1]
#                 pred = Annotation(
#                     id=example["id"],
#                     entity_type=id_to_type[type_id],
#                     start_char=start_char,
#                     end_char=end_char,
#                     text=example["text"][start_char:end_char],
#                 )
#                 score = float(joint_logit[type_id, start_index, end_index])
#                 if pred in example_pred_scores:
#                     example_pred_scores[pred] = max(example_pred_scores[pred], score)
#                 else:
#                     example_pred_scores[pred] = score

#         # keep your existing overlap resolver (consider making it type-aware if needed)
#         example_predictions = remove_overlaps(example_pred_scores)

#         # metrics aggregation
#         for t in metrics_by_type.keys():
#             for k, v in compute_tp_fn_fp(
#                 example_predictions if t == "all" else set(filter(lambda x: x.entity_type == t, example_predictions)),
#                 example_annotations if t == "all" else set(filter(lambda x: x.entity_type == t, example_annotations)),
#             ).items():
#                 metrics_by_type[t][k] += v

#         all_annotations.update(example_annotations)
#         all_predictions.update(example_predictions)

#         # start/end diagnostics
#         example_gold_starts = set((x.entity_type, x.start_char) for x in example_annotations)
#         example_pred_starts = set((x.entity_type, x.start_char) for x in example_predictions)
#         for t in start_metrics_by_type.keys():
#             for k, v in compute_tp_fn_fp(
#                 example_pred_starts if t == "all" else set(filter(lambda x: x[0] == t, example_pred_starts)),
#                 example_gold_starts if t == "all" else set(filter(lambda x: x[0] == t, example_gold_starts)),
#             ).items():
#                 start_metrics_by_type[t][k] += v

#         example_gold_ends = set((x.entity_type, x.end_char) for x in example_annotations)
#         example_pred_ends = set((x.entity_type, x.end_char) for x in example_predictions)
#         for t in end_metrics_by_type.keys():
#             for k, v in compute_tp_fn_fp(
#                 example_pred_ends if t == "all" else set(filter(lambda x: x[0] == t, example_pred_ends)),
#                 example_gold_ends if t == "all" else set(filter(lambda x: x[0] == t, example_gold_ends)),
#             ).items():
#                 end_metrics_by_type[t][k] += v

#     # summarize metrics
#     metrics = collections.OrderedDict()
#     precisions, recalls = [], []
#     sorted_entity_types = ["all"] + sorted(entity_type_vocab, key=lambda x: entity_type_count[x], reverse=True)
#     for x, x_metrics_by_type in {"span": metrics_by_type, "start": start_metrics_by_type, "end": end_metrics_by_type}.items():
#         metrics[x] = {}
#         for t in sorted_entity_types:
#             metrics_for_t = compute_precision_recall_f1(**x_metrics_by_type[t])
#             f1, precision, recall = metrics_for_t["f1"], metrics_for_t["precision"], metrics_for_t["recall"]
#             if x == "span" and t != "all":
#                 precisions.append(precision)
#                 recalls.append(recall)
#             metrics[x][t] = {k: v for k, v in metrics_for_t.items()}

#     for t in sorted_entity_types:
#         support = entity_type_count[t]
#         logger.info(f"***** {t} ({support}) *****")
#         for x in metrics:
#             f1, precision, recall = metrics[x][t]["f1"], metrics[x][t]["precision"], metrics[x][t]["recall"]
#             logger.info(f"F1 = {f1:>6.1%}, Precision = {precision:>6.1%}, Recall = {recall:>6.1%} (for {x})")

#     # Save outputs
#     if output_dir is not None:
#         if not os.path.isdir(output_dir):
#             raise EnvironmentError(f"{output_dir} is not a directory.")

#         example_id_to_predictions = {}
#         for pred in all_predictions:
#             example_id_to_predictions.setdefault(pred.id, set()).add(
#                 (pred.start_char, pred.end_char, pred.entity_type, pred.text)
#             )

#         predictions_to_save = []
#         for example in examples:
#             example = copy.deepcopy(example)
#             example.pop("word_start_chars")
#             example.pop("word_end_chars")
#             gold_ner = set()
#             for entity_type, start_char, end_char in zip(
#                 example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
#                 gold_ner.add((start_char, end_char, entity_type, example["text"][start_char:end_char]))
#             example.pop("entity_types")
#             example.pop("entity_start_chars")
#             example.pop("entity_end_chars")

#             pred_ner = example_id_to_predictions.get(example["id"], set())
#             example["gold_ner"] = sorted(gold_ner)
#             example["pred_ner"] = sorted(pred_ner)
#             predictions_to_save.append(example)

#         prediction_file = os.path.join(output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json")
#         logger.info(f"Saving predictions to {prediction_file}.")
#         with open(prediction_file, "w") as writer:
#             for pred in predictions_to_save:
#                 writer.write(json.dumps(pred) + "\n")

#         metric_file = os.path.join(output_dir, "metrics.json" if prefix is None else f"{prefix}_metrics.json")
#         logger.info(f"Saving metrics to {metric_file}.")
#         with open(metric_file, "a") as writer:
#             writer.write(json.dumps(metrics) + "\n")

#     metrics_summary = {
#         "f1": metrics["span"]["all"]["f1"],
#         "precision": metrics["span"]["all"]["precision"],
#         "recall": metrics["span"]["all"]["recall"]
#     }

#     return {
#         "predictions": all_predictions,
#         "labels": all_annotations,
#         "metrics": metrics_summary,
#     }


# def postprocess_flat_bce_predictions(
#     examples,
#     features,
#     predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
#     id_to_type: List[str],
#     max_span_length: int = 30,
#     output_dir: Optional[str] = None,
#     prefix: Optional[str] = None,
#     log_level: Optional[int] = logging.WARNING,
#     tokenizer=None,
#     **kwargs,
# ) -> Dict:
#     logger.setLevel(log_level)

#     if len(predictions) != 3:
#         raise ValueError("`predictions` should be a tuple with three elements (start_logits, end_logits, span_logits).")
#     all_start_logits, all_end_logits, all_span_logits = predictions

#     if len(predictions[0]) != len(features):
#         raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

#     # ---- kwargs (sane defaults) ----
#     temp = float(kwargs.get("temp", 1.0))                          # temperature scaling
#     start_threshold = float(kwargs.get("start_threshold", 0.5))     # gate on start prob
#     end_threshold   = float(kwargs.get("end_threshold", 0.5))       # gate on end prob
#     span_threshold  = kwargs.get("span_threshold", 0.5)             # prob threshold; float or {type: thr}
#     alpha = float(kwargs.get("alpha", 0.7))                         # start logit weight in fusion
#     beta  = float(kwargs.get("beta", 0.7))                          # end logit weight in fusion
#     localmax_window = int(kwargs.get("localmax_window", 1))         # 0 disables local-max gating
#     span_top_k = kwargs.get("span_top_k", None)                     # None disables; else int
#     use_fusion = bool(kwargs.get("use_fusion", True))               # allow turning fusion off

#     # Build a map example -> its features
#     example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
#     features_per_example = collections.defaultdict(list)
#     for i, feature in enumerate(features):
#         features_per_example[example_id_to_index[feature["example_id"]]].append(i)

#     # Gold / predicted sets and counters
#     all_annotations: Set[Annotation] = set()
#     all_predictions: Set[Annotation] = set()

#     entity_type_vocab = list(set(id_to_type))
#     entity_type_count = collections.defaultdict(int)
#     metrics_by_type = {t: {"tp": 0, "fn": 0, "fp": 0} for t in entity_type_vocab + ["all"]}
#     start_metrics_by_type = {t: {"tp": 0, "fn": 0, "fp": 0} for t in entity_type_vocab + ["all"]}
#     end_metrics_by_type   = {t: {"tp": 0, "fn": 0, "fp": 0} for t in entity_type_vocab + ["all"]}

#     logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

#     # Helper: local-max gate along axis=1 (length L)
#     def _localmax_1d(prob_2d: np.ndarray, w: int) -> np.ndarray:
#         # prob_2d: [T, L] → bool peaks [T, L]
#         if w <= 0:
#             return np.ones_like(prob_2d, dtype=bool)
#         try:
#             from numpy.lib.stride_tricks import sliding_window_view
#             pad = np.pad(prob_2d, ((0, 0), (w, w)), mode="edge")     # [T, L+2w]
#             win = sliding_window_view(pad, window_shape=(2*w+1), axis=1)  # [T, L, 2w+1]
#             return prob_2d == win.max(axis=-1)
#         except Exception:
#             # Fallback: simple neighborhood check
#             T, L = prob_2d.shape
#             out = np.ones((T, L), dtype=bool)
#             for d in range(1, w+1):
#                 out &= (prob_2d >= np.roll(prob_2d, +d, axis=1))
#                 out &= (prob_2d >= np.roll(prob_2d, -d, axis=1))
#             return out

#     # Loop examples
#     for example_index, example in enumerate(examples):
#         example_annotations: Set[Annotation] = set()
#         example_pred_scores: Dict[Annotation, float] = {}

#         # gold annotations
#         for entity_type, start_char, end_char in zip(
#             example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]
#         ):
#             entity_type_count["all"] += 1
#             entity_type_count[entity_type] += 1
#             example_annotations.add(Annotation(
#                 id=example["id"],
#                 entity_type=entity_type,
#                 start_char=start_char,
#                 end_char=end_char,
#                 text=example["text"][start_char:end_char],
#             ))

#         # features for this example
#         for feature_index in features_per_example[example_index]:
#             # masks
#             token_start_mask = np.asarray(features[feature_index]["token_start_mask"], dtype=bool)  # [L]
#             token_end_mask   = np.asarray(features[feature_index]["token_end_mask"],   dtype=bool)  # [L]
#             default_span_mask = np.asarray(features[feature_index]["default_span_mask"], dtype=bool)  # [L,L]
#             offset_mapping = features[feature_index]["offset_mapping"]

#             # logits → float32 & temperature scale
#             start_logits = all_start_logits[feature_index].astype(np.float32) / temp   # [T,L]
#             end_logits   = all_end_logits[feature_index].astype(np.float32)   / temp   # [T,L]
#             span_logits  = all_span_logits[feature_index].astype(np.float32)  / temp   # [T,L,L]

#             # probs for gating (keep logits for ranking)
#             start_prob = 1.0 / (1.0 + np.exp(-start_logits))
#             end_prob   = 1.0 / (1.0 + np.exp(-end_logits))
#             span_prob  = 1.0 / (1.0 + np.exp(-span_logits))

#             T = span_prob.shape[0]
#             L = span_prob.shape[-1]

#             # validity masks
#             i_idx = np.arange(L)[:, None]
#             j_idx = np.arange(L)[None, :]
#             length_mask = ((j_idx - i_idx + 1) <= max_span_length)[None, :, :]
#             valid = (
#                 token_start_mask[None, :, None] &
#                 token_end_mask[None, None, :] &
#                 default_span_mask[None, :, :] &
#                 length_mask
#             )

#             # local-max gating on start/end
#             start_is_peak = _localmax_1d(start_prob, localmax_window)   # [T,L]
#             end_is_peak   = _localmax_1d(end_prob,   localmax_window)   # [T,L]

#             # thresholds (allow per-type dict for span)
#             if isinstance(span_threshold, dict):
#                 thr = np.array([span_threshold.get(id_to_type[t], 0.5) for t in range(T)], dtype=np.float32)[:, None, None]
#             else:
#                 thr = float(span_threshold)

#             gate = (
#                 (start_prob >= start_threshold)[:, :, None] &
#                 (end_prob   >= end_threshold)[:, None, :] &
#                 start_is_peak[:, :, None] &
#                 end_is_peak[:, None, :] &
#                 (span_prob  >= thr)
#             )

#             # fused joint score in logit space (better ranking)
#             if use_fusion:
#                 joint_logit = span_logits + alpha * start_logits[:, :, None] + beta * end_logits[:, None, :]
#                 score_for_ranking = joint_logit
#             else:
#                 score_for_ranking = span_logits  # fallback

#             keep = gate & valid  # final boolean mask

#             # optional: per-type top-K after gating
#             if isinstance(span_top_k, int) and span_top_k > 0:
#                 keep_flat = keep.reshape(T, -1)
#                 score_flat = score_for_ranking.reshape(T, -1)
#                 new_keep = np.zeros_like(keep_flat, dtype=bool)
#                 for t in range(T):
#                     idx = np.where(keep_flat[t])[0]
#                     if idx.size == 0:
#                         continue
#                     k = min(span_top_k, idx.size)
#                     # take top-k by fused score
#                     sel = idx[np.argpartition(-score_flat[t, idx], kth=k-1)[:k]]
#                     new_keep[t, sel] = True
#                 keep = new_keep.reshape(keep.shape)

#             type_ids, start_indexes, end_indexes = keep.nonzero()

#             # map back to chars
#             for type_id, start_index, end_index in zip(type_ids, start_indexes, end_indexes):
#                 if (
#                     start_index >= len(offset_mapping)
#                     or end_index >= len(offset_mapping)
#                     or offset_mapping[start_index] is None
#                     or offset_mapping[end_index] is None
#                 ):
#                     continue
#                 if end_index - start_index + 1 > max_span_length:
#                     continue
#                 start_char, end_char = offset_mapping[start_index][0], offset_mapping[end_index][1]
#                 pred = Annotation(
#                     id=example["id"],
#                     entity_type=id_to_type[type_id],
#                     start_char=start_char,
#                     end_char=end_char,
#                     text=example["text"][start_char:end_char],
#                 )
#                 score = float(score_for_ranking[type_id, start_index, end_index])
#                 if pred in example_pred_scores:
#                     example_pred_scores[pred] = max(example_pred_scores[pred], score)
#                 else:
#                     example_pred_scores[pred] = score

#         # non-maximum suppression / overlap resolver (type-aware recommended)
#         example_predictions = remove_overlaps(example_pred_scores)

#         # metrics aggregation
#         for t in metrics_by_type.keys():
#             for k, v in compute_tp_fn_fp(
#                 example_predictions if t == "all" else set(filter(lambda x: x.entity_type == t, example_predictions)),
#                 example_annotations if t == "all" else set(filter(lambda x: x.entity_type == t, example_annotations)),
#             ).items():
#                 metrics_by_type[t][k] += v

#         all_annotations.update(example_annotations)
#         all_predictions.update(example_predictions)

#         # start/end diagnostic metrics
#         example_gold_starts = set((x.entity_type, x.start_char) for x in example_annotations)
#         example_pred_starts = set((x.entity_type, x.start_char) for x in example_predictions)
#         for t in start_metrics_by_type.keys():
#             for k, v in compute_tp_fn_fp(
#                 example_pred_starts if t == "all" else set(filter(lambda x: x[0] == t, example_pred_starts)),
#                 example_gold_starts if t == "all" else set(filter(lambda x: x[0] == t, example_gold_starts)),
#             ).items():
#                 start_metrics_by_type[t][k] += v

#         example_gold_ends = set((x.entity_type, x.end_char) for x in example_annotations)
#         example_pred_ends = set((x.entity_type, x.end_char) for x in example_predictions)
#         for t in end_metrics_by_type.keys():
#             for k, v in compute_tp_fn_fp(
#                 example_pred_ends if t == "all" else set(filter(lambda x: x[0] == t, example_pred_ends)),
#                 example_gold_ends if t == "all" else set(filter(lambda x: x[0] == t, example_gold_ends)),
#             ).items():
#                 end_metrics_by_type[t][k] += v

#     # summarize metrics
#     metrics = collections.OrderedDict()
#     precisions, recalls = [], []
#     sorted_entity_types = ["all"] + sorted(entity_type_vocab, key=lambda x: entity_type_count[x], reverse=True)
#     for x, x_metrics_by_type in {"span": metrics_by_type, "start": start_metrics_by_type, "end": end_metrics_by_type}.items():
#         metrics[x] = {}
#         for t in sorted_entity_types:
#             mr = compute_precision_recall_f1(**x_metrics_by_type[t])
#             f1, precision, recall = mr["f1"], mr["precision"], mr["recall"]
#             if x == "span" and t != "all":
#                 precisions.append(precision); recalls.append(recall)
#             metrics[x][t] = {k: v for k, v in mr.items()}

#     for t in sorted_entity_types:
#         support = entity_type_count[t]
#         logger.info(f"***** {t} ({support}) *****")
#         for x in metrics:
#             f1, precision, recall = metrics[x][t]["f1"], metrics[x][t]["precision"], metrics[x][t]["recall"]
#             logger.info(f"F1 = {f1:>6.1%}, Precision = {precision:>6.1%}, Recall = {recall:>6.1%} (for {x})")

#     # Save outputs
#     if output_dir is not None:
#         if not os.path.isdir(output_dir):
#             raise EnvironmentError(f"{output_dir} is not a directory.")

#         example_id_to_predictions = {}
#         for pred in all_predictions:
#             example_id_to_predictions.setdefault(pred.id, set()).add(
#                 (pred.start_char, pred.end_char, pred.entity_type, pred.text)
#             )

#         predictions_to_save = []
#         for example in examples:
#             example = copy.deepcopy(example)
#             for k in ["word_start_chars", "word_end_chars", "ner_tags_idx", "ner_tags_str"]:
#                 if k in example:
#                     example.pop(k, None)
#             gold_ner = set()
#             for et, sc, ec in zip(example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
#                 gold_ner.add((sc, ec, et, example["text"][sc:ec]))
#             example.pop("entity_types"); example.pop("entity_start_chars"); example.pop("entity_end_chars")
#             pred_ner = example_id_to_predictions.get(example["id"], set())
#             example["gold_ner"] = sorted(gold_ner)
#             example["pred_ner"] = sorted(pred_ner)
#             predictions_to_save.append(example)

#         prediction_file = os.path.join(output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json")
#         logger.info(f"Saving predictions to {prediction_file}.")
#         with open(prediction_file, "w") as writer:
#             for pred in predictions_to_save:
#                 writer.write(json.dumps(pred) + "\n")

#         metric_file = os.path.join(output_dir, "metrics.json" if prefix is None else f"{prefix}_metrics.json")
#         logger.info(f"Saving metrics to {metric_file}.")
#         with open(metric_file, "a") as writer:
#             writer.write(json.dumps(metrics) + "\n")

#     metrics_summary = {
#         "f1": metrics["span"]["all"]["f1"],
#         "precision": metrics["span"]["all"]["precision"],
#         "recall": metrics["span"]["all"]["recall"],
#     }

#     return {
#         "predictions": all_predictions,
#         "labels": all_annotations,
#         "metrics": metrics_summary,
#     }


# def postprocess_flat_bce_predictions(
#     examples,
#     features,
#     predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
#     id_to_type: List[str],
#     max_span_length: int = 30,
#     output_dir: Optional[str] = None,
#     prefix: Optional[str] = None,
#     log_level: Optional[int] = logging.WARNING,
#     tokenizer = None,
#     **kwargs,
# ) -> Dict:
#     logger.setLevel(log_level)

#     if len(predictions) != 3:
#         raise ValueError("`predictions` should be a tuple with three elements (start_logits, end_logits, span_logits).")
#     all_start_logits, all_end_logits, all_span_logits = predictions

#     if len(predictions[0]) != len(features):
#         raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

#     # Build a map example to its corresponding features.
#     example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
#     features_per_example = collections.defaultdict(list)
#     for i, feature in enumerate(features):
#         features_per_example[example_id_to_index[feature["example_id"]]].append(i)

#     # The gold annotations.
#     all_annotations = set()
#     # The dictionaries we have to fill.
#     all_predictions = set()

#     entity_type_vocab = list(set(id_to_type))
#     entity_type_count = collections.defaultdict(int)
#     metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}
#     start_metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}
#     end_metrics_by_type = {entity_type: {"tp": 0, "fn": 0, "fp": 0} for entity_type in entity_type_vocab + ["all"]}

#     # Logging.
#     logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

#     # Let's loop over all the examples!
#     for example_index, example in enumerate(examples):
#         example_annotations = set()
#         example_pred_scores = {}

#         # Looping through all NER annotations.
#         for entity_type, start_char, end_char in zip(
#             example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
#             entity_type_count["all"] += 1
#             entity_type_count[entity_type] += 1
#             example_annotations.add(Annotation(
#                 id=example["id"],
#                 entity_type=entity_type,
#                 start_char=start_char,
#                 end_char=end_char,
#                 text=example["text"][start_char:end_char]
#             ))

#         # Those are the indices of the features associated to the current example.
#         feature_indices = features_per_example[example_index]

#         # Looping through all the features associated to the current example.
#         for feature_index in feature_indices:
#             # We grab the masks for start and end indices.
#             token_start_mask = np.array(features[feature_index]["token_start_mask"]).astype(bool)
#             token_end_mask = np.array(features[feature_index]["token_end_mask"]).astype(bool)

#             # --- get logits and convert to probabilities ---
#             span_logits = all_span_logits[feature_index].astype(np.float32)     # [T, L, L]
#             span_probs  = 1.0 / (1.0 + np.exp(-span_logits))                    # sigmoid

#             # --- validity mask (where we are allowed to predict) ---
#             L = span_probs.shape[-1]

#             # If you stored default_span_mask in features (recommended):
#             default_span_mask = np.array(features[feature_index]["default_span_mask"]).astype(bool)  # [L, L]
#             # else, fall back to: allow only i<=j (upper triangle incl. diag)
#             # default_span_mask = np.triu(np.ones((L, L), dtype=bool), k=0)

#             valid_mask = (
#                 token_start_mask[np.newaxis, :, np.newaxis] &    # valid start token
#                 token_end_mask[np.newaxis, np.newaxis, :] &      # valid end token
#                 default_span_mask[np.newaxis, :, :]              # legal spans (padding, i<=j, etc.)
#             )

#             # --- max span length mask ---
#             # length(i,j) = j - i + 1
#             i_idx = np.arange(L).reshape(L, 1)
#             j_idx = np.arange(L).reshape(1, L)
#             length_mask = ((j_idx - i_idx + 1) <= max_span_length)[np.newaxis, :, :]

#             # --- final selection by probability threshold ---
#             # allow per-type or global threshold; support kwargs['span_threshold'] or dict
#             span_threshold = kwargs.get("span_threshold", 0.5)
#             if isinstance(span_threshold, dict):
#                 # build per-type thresholds array
#                 thr = np.array([span_threshold.get(id_to_type[t], 0.5) for t in range(span_probs.shape[0])], dtype=np.float32)
#                 thr = thr[:, None, None]  # [T,1,1]
#             else:
#                 thr = float(span_threshold)

#             span_preds = (span_probs >= thr) & valid_mask & length_mask

#             # optional: keep only top-K spans per type to control explosion
#             top_k = kwargs.get("span_top_k", None)
#             if top_k is not None:
#                 # flatten [T,L,L], keep top_k per type by prob, then re-mask
#                 flat = span_probs.reshape(span_probs.shape[0], -1)
#                 idx = np.argpartition(-flat, kth=min(top_k-1, flat.shape[1]-1), axis=1)[:, :top_k]
#                 top_mask = np.zeros_like(flat, dtype=bool)
#                 row_idx = np.arange(flat.shape[0])[:, None]
#                 top_mask[row_idx, idx] = True
#                 top_mask = top_mask.reshape(span_probs.shape)
#                 span_preds = span_preds & top_mask

#             type_ids, start_indexes, end_indexes = span_preds.nonzero()

#             # This is what will allow us to map some the positions in our logits to span of texts in the original context.
#             offset_mapping = features[feature_index]["offset_mapping"]

#             # Go through all start and end indices.
#             for type_id, start_index, end_index in zip(type_ids, start_indexes, end_indexes):
#                 # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
#                 # to part of the input_ids that are not in the context.
#                 if (
#                     start_index >= len(offset_mapping)
#                     or end_index >= len(offset_mapping)
#                     or offset_mapping[start_index] is None
#                     or offset_mapping[end_index] is None
#                 ):
#                     continue
#                 # Don't consider spans with a length that is > max_span_length.
#                 if end_index - start_index + 1 > max_span_length:
#                     continue
#                 # A prediction contains (example_id, entity_type, start_index, end_index)
#                 start_char, end_char = offset_mapping[start_index][0], offset_mapping[end_index][1]
#                 pred = Annotation(
#                     id=example["id"],
#                     entity_type=id_to_type[type_id],
#                     start_char=start_char,
#                     end_char=end_char,
#                     text=example["text"][start_char:end_char],
#                 )
#                 if pred in example_pred_scores:
#                     example_pred_scores[pred] = max(example_pred_scores[pred], span_logits[type_id][start_index][end_index])
#                 else:
#                     example_pred_scores[pred] = span_logits[type_id][start_index][end_index]

#         example_predictions = remove_overlaps(example_pred_scores)

#         for t in metrics_by_type.keys():
#             for k, v in compute_tp_fn_fp(
#                 example_predictions if t == "all" else set(filter(lambda x: x.entity_type == t, example_predictions)),
#                 example_annotations if t == "all" else set(filter(lambda x: x.entity_type == t, example_annotations)),
#             ).items():
#                 metrics_by_type[t][k] += v

#         all_annotations.update(example_annotations)
#         all_predictions.update(example_predictions)

#         example_gold_starts = set((x.entity_type, x.start_char) for x in example_annotations)
#         example_pred_starts = set((x.entity_type, x.start_char) for x in example_predictions)
#         for t in start_metrics_by_type.keys():
#             for k, v in compute_tp_fn_fp(
#                 example_pred_starts if t == "all" else set(filter(lambda x: x[0] == t, example_pred_starts)),
#                 example_gold_starts if t == "all" else set(filter(lambda x: x[0] == t, example_gold_starts)),
#             ).items():
#                 start_metrics_by_type[t][k] += v

#         example_gold_ends = set((x.entity_type, x.end_char) for x in example_annotations)
#         example_pred_ends = set((x.entity_type, x.end_char) for x in example_predictions)
#         for t in end_metrics_by_type.keys():
#             for k, v in compute_tp_fn_fp(
#                 example_pred_ends if t == "all" else set(filter(lambda x: x[0] == t, example_pred_ends)),
#                 example_gold_ends if t == "all" else set(filter(lambda x: x[0] == t, example_gold_ends)),
#             ).items():
#                 end_metrics_by_type[t][k] += v

#     metrics = collections.OrderedDict()
#     precisions, recalls = [], []
#     sorted_entity_types = ["all"] + sorted(entity_type_vocab, key=lambda x: entity_type_count[x], reverse=True)
#     for x, x_metrics_by_type in {"span": metrics_by_type, "start": start_metrics_by_type, "end": end_metrics_by_type}.items():
#         metrics[x] = {}
#         for t in sorted_entity_types:
#             metrics_for_t = compute_precision_recall_f1(**x_metrics_by_type[t])
#             f1, precision, recall = metrics_for_t["f1"], metrics_for_t["precision"], metrics_for_t["recall"]
#             if x == "span" and t != "all":
#                 precisions.append(precision)
#                 recalls.append(recall)
#             metrics[x][t] = {}
#             for k, v in metrics_for_t.items():
#                 metrics[x][t][k] = v

#     for t in sorted_entity_types:
#         support = entity_type_count[t]
#         logger.info(f"***** {t} ({support}) *****")
#         for x in metrics:
#             f1, precision, recall = metrics[x][t]["f1"], metrics[x][t]["precision"], metrics[x][t]["recall"]
#             logger.info(f"F1 = {f1:>6.1%}, Precision = {precision:>6.1%}, Recall = {recall:>6.1%} (for {x})")

#     # If we have an output_dir, let's save all those dicts.
#     if output_dir is not None:
#         if not os.path.isdir(output_dir):
#             raise EnvironmentError(f"{output_dir} is not a directory.")

#         # Convert flat predictions to hierarchical.
#         example_id_to_predictions = {}
#         for pred in all_predictions:
#             example_id = pred.id
#             if example_id not in example_id_to_predictions:
#                 example_id_to_predictions[example_id] = set()
#             example_id_to_predictions[example_id].add((pred.start_char, pred.end_char, pred.entity_type, pred.text))

#         predictions_to_save = []
#         for example in examples:
#             example = copy.deepcopy(example)
#             example.pop("word_start_chars")
#             example.pop("word_end_chars")
#             if "ner_tags_idx" in example:
#                 example.pop("ner_tags_idx")
#                 example.pop("ner_tags_str")
#             gold_ner = set()
#             for entity_type, start_char, end_char in zip(
#                 example["entity_types"], example["entity_start_chars"], example["entity_end_chars"]):
#                 gold_ner.add((start_char, end_char, entity_type, example["text"][start_char:end_char]))
#             example.pop("entity_types")
#             example.pop("entity_start_chars")
#             example.pop("entity_end_chars")

#             pred_ner = example_id_to_predictions.get(example["id"], set())
#             example["gold_ner"] = sorted(gold_ner)
#             example["pred_ner"] = sorted(pred_ner)

#             predictions_to_save.append(example)

#         prediction_file = os.path.join(
#             output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
#         )

#         logger.info(f"Saving predictions to {prediction_file}.")
#         with open(prediction_file, "w") as writer:
#             for pred in predictions_to_save:
#                 writer.write(json.dumps(pred) + "\n")

#         metric_file = os.path.join(
#             output_dir, "metrics.json" if prefix is None else f"{prefix}_metrics.json"
#         )

#         logger.info(f"Saving metrics to {metric_file}.")
#         with open(metric_file, "a") as writer:
#             writer.write(json.dumps(metrics) + "\n")

#     metrics = {
#         "f1": metrics["span"]["all"]["f1"],
#         "precision": metrics["span"]["all"]["precision"],
#         "recall": metrics["span"]["all"]["recall"]
#     }

#     return {
#         "predictions": all_predictions,
#         "labels": all_annotations,
#         "metrics": metrics,
#     }