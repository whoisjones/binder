from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from transformers import PreTrainedTokenizer
from src.utils import Span

@dataclass
class BinderDataCollator:
    tokenizer: PreTrainedTokenizer
    id2label: dict[str, dict[int, str]]

    def __call__(self, features: List) -> Dict[str, Any]:
        batch = {}
        stage = features[0]['split']

        batch['input_ids'] = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        batch['attention_mask'] = torch.tensor([f['attention_mask'] for f in features], dtype=torch.bool)
        if "token_type_ids" in features[0]:
            batch['token_type_ids'] = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)

        if stage == 'train':
            batch_labels = set([ann['type'] for feature in features for ann in feature['ner']])
            batch_id2label = {i: label for i, label in enumerate(batch_labels)}
            batch_label2id = {label: i for i, label in enumerate(batch_labels)}
        elif stage == 'eval' or stage == 'predict':
            batch_id2label = self.id2label[stage]
            batch_label2id = {v: k for k, v in batch_id2label.items()}
        else:
            raise ValueError(f"Unknown stage: {stage}")

        tokenized_labels = self.tokenizer(
            list(batch_id2label.values()),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch['type_input_ids'] = tokenized_labels['input_ids']
        batch['type_attention_mask'] = tokenized_labels['attention_mask']
        if 'token_type_ids' in tokenized_labels:
            batch['type_token_type_ids'] = tokenized_labels['token_type_ids']

        if stage == 'train':
            annotations = []
            for feature in features:
                token_start_mask = feature['token_start_mask']
                token_end_mask = feature['token_end_mask']
                default_span_mask = feature['default_span_mask']

                start_negative_mask = [token_start_mask[:] for _ in batch_id2label]
                end_negative_mask = [token_end_mask[:] for _ in batch_id2label]
                span_negative_mask = [[x[:] for x in default_span_mask] for _ in batch_id2label]

                # Exclude the start/end of the NER span.
                for ann in feature['ner']:
                    start_negative_mask[batch_label2id[ann['type']]][ann['start']] = 0
                    end_negative_mask[batch_label2id[ann['type']]][ann['end']] = 0
                    span_negative_mask[batch_label2id[ann['type']]][ann['start']][ann['end']] = 0

                annotations.append({
                    "annotations": feature['ner'],
                    "start_negative_mask": start_negative_mask,
                    "end_negative_mask": end_negative_mask,
                    "span_negative_mask": span_negative_mask,
                })
            

            # For training
            ner = {}
            # Collate negative mask with shape [batch_size, num_types, ...].
            start_negative_mask, end_negative_mask, span_negative_mask = [], [], []
            # [batch_size, num_types, seq_length]
            start_negative_mask = torch.tensor([a["start_negative_mask"] for a in annotations], dtype=torch.bool)
            end_negative_mask = torch.tensor([a["end_negative_mask"] for a in annotations], dtype=torch.bool)
            # [batch_size, num_types, seq_length, seq_length]
            span_negative_mask = torch.tensor([a["span_negative_mask"] for a in annotations], dtype=torch.bool)
            # Include [CLS]
            start_negative_mask[:, :, 0] = 1
            end_negative_mask[:, :, 0] = 1
            span_negative_mask[:, :, 0, 0] = 1

            ner['start_negative_mask'] =  start_negative_mask
            ner['end_negative_mask'] = end_negative_mask
            ner['span_negative_mask'] = span_negative_mask

            # Collate mention span examples.
            feature_spans = []
            for feature_id, feature in enumerate(features):
                spans = []
                for ann in feature["ner"]:
                    type_id, start, end = batch_label2id[ann["type"]], ann["start"], ann["end"]

                    start_mask = start_negative_mask[feature_id][type_id].detach().clone()
                    start_mask[start] = 1

                    end_mask = end_negative_mask[feature_id][type_id].detach().clone()
                    end_mask[end] = 1

                    span_mask = span_negative_mask[feature_id][type_id].detach().clone()
                    span_mask[start][end] = 1

                    spans.append(
                        Span(type_id, start, end, start_mask, end_mask, span_mask)
                    )
                feature_spans.append(spans)

            feature_ids = []
            for feature_id, spans in enumerate(feature_spans):
                feature_ids += [feature_id] * len(spans)
            span_type_ids = [s.type_id for spans in feature_spans for s in spans]

            ner["example_indices"] = [feature_ids, span_type_ids]
            # [batch_size]
            ner["example_starts"] = [s.start for spans in feature_spans for s in spans]
            ner["example_ends"] = [s.end for spans in feature_spans for s in spans]
            # [batch_size, seq_length]
            ner["example_start_masks"] = torch.stack([s.start_mask for spans in feature_spans for s in spans])
            ner["example_end_masks"] = torch.stack([s.end_mask for spans in feature_spans for s in spans])
            # [batch_size, seq_length, seq_length]
            ner["example_span_masks"] = torch.stack([s.span_mask for spans in feature_spans for s in spans])

            batch['ner'] = ner

        return batch

@dataclass
class BinderFocalDataCollator:
    type_tokenizer: PreTrainedTokenizer
    id2label: dict[str, dict[int, str]]

    def __call__(self, features: List) -> Dict[str, Any]:
        batch = {}
        stage = features[0]['split']

        batch['input_ids'] = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        batch['attention_mask'] = torch.tensor([f['attention_mask'] for f in features], dtype=torch.bool)
        if "token_type_ids" in features[0]:
            batch['token_type_ids'] = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)

        if stage == 'train':
            batch_labels = set([ann['type'] for feature in features for ann in feature['ner']])
            batch_id2label = {i: label for i, label in enumerate(batch_labels)}
            batch_label2id = {label: i for i, label in enumerate(batch_labels)}
        elif stage == 'eval' or stage == 'predict':
            batch_id2label = self.id2label[stage]
            batch_label2id = {v: k for k, v in batch_id2label.items()}
        else:
            raise ValueError(f"Unknown stage: {stage}")

        tokenized_labels = self.type_tokenizer(
            list(batch_id2label.values()),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch['type_input_ids'] = tokenized_labels['input_ids']
        batch['type_attention_mask'] = tokenized_labels['attention_mask']
        if 'token_type_ids' in tokenized_labels:
            batch['type_token_type_ids'] = tokenized_labels['token_type_ids']
        

        if stage == 'train':
            B = len(features); T = len(batch_id2label); L = len(features[0]['token_start_mask'])
            start_valid_mask = torch.zeros((B, T, L), dtype=torch.bool)
            end_valid_mask   = torch.zeros((B, T, L), dtype=torch.bool)
            span_valid_mask  = torch.zeros((B, T, L, L), dtype=torch.bool)

            start_targets = torch.zeros((B, T, L), dtype=torch.float32)
            end_targets   = torch.zeros((B, T, L), dtype=torch.float32)
            span_targets  = torch.zeros((B, T, L, L), dtype=torch.float32)

            for b, feat in enumerate(features):
                token_start_mask = torch.tensor(feat['token_start_mask'], dtype=torch.bool)     # True where valid token (not pad)
                token_end_mask   = torch.tensor(feat['token_end_mask'], dtype=torch.bool)
                default_span_mask = torch.tensor(feat['default_span_mask'], dtype=torch.bool)   # [L, L], True where valid (i<=j etc.)

                # validity (same for every type in this example)
                start_valid_mask[b, :, :] = token_start_mask.unsqueeze(0).expand(T, -1)
                end_valid_mask[b, :, :] = token_end_mask  .unsqueeze(0).expand(T, -1)
                span_valid_mask[b, :, :, :] = default_span_mask.unsqueeze(0).expand(T, -1, -1)

                # positives
                for ann in feat['ner']:
                    t = batch_label2id[ann['type']]
                    i, j = ann['start'], ann['end']
                    start_targets[b, t, i] = 1.0
                    end_targets[b, t, j]   = 1.0
                    span_targets[b, t, i, j] = 1.0
            
            batch['ner'] = {
                "start_targets": start_targets,
                "end_targets": end_targets,
                "span_targets": span_targets,
                "start_valid_mask": start_valid_mask,
                "end_valid_mask": end_valid_mask,
                "span_valid_mask": span_valid_mask,
            }

        return batch