from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel, AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.file_utils import ModelOutput
from .config import BinderConfig, BinderFocalConfig


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def masked_log_softmax(vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1) -> torch.Tensor:
    """
    `torch.nn.functional.log_softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a log_softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular log_softmax.
    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not `nan`.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you `nans`.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.
        vector = vector + (mask + tiny_value_of_dtype(vector.dtype)).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def contrastive_loss(
    scores: torch.FloatTensor,
    positions: Union[List[int], Tuple[List[int], List[int]]],
    mask: torch.BoolTensor,
    prob_mask: torch.BoolTensor = None,
) -> torch.FloatTensor:
    batch_size, seq_length = scores.size(0), scores.size(1)
    if len(scores.shape) == 3:
        scores = scores.view(batch_size, -1)
        mask = mask.view(batch_size, -1)
        log_probs = masked_log_softmax(scores, mask)
        log_probs = log_probs.view(batch_size, seq_length, seq_length)
        start_positions, end_positions = positions
        batch_indices = list(range(batch_size))
        log_probs = log_probs[batch_indices, start_positions, end_positions]
    else:
        log_probs = masked_log_softmax(scores, mask)
        batch_indices = list(range(batch_size))
        log_probs = log_probs[batch_indices, positions]
    if prob_mask is not None:
        log_probs = log_probs * prob_mask
    return - log_probs.mean()


@dataclass
class BinderModelOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    start_scores: torch.FloatTensor = None
    end_scores: torch.FloatTensor = None
    span_scores: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Binder(PreTrainedModel):
    config_class = BinderConfig

    def __init__(self, config):
        super().__init__(config)

        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            cache_dir=config.cache_dir,
            revision=config.revision,
            use_auth_token=config.use_auth_token,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )
        self.hf_config = hf_config
        self.config.pruned_heads = hf_config.pruned_heads
        self.dropout = torch.nn.Dropout(hf_config.hidden_dropout_prob)
        self.type_start_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        self.type_end_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        self.type_span_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        self.start_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        self.end_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        if config.use_span_width_embedding:
            self.span_linear = torch.nn.Linear(hf_config.hidden_size * 2 + config.linear_size, config.linear_size)
            self.width_embeddings = torch.nn.Embedding(config.max_span_width, config.linear_size, padding_idx=0)
        else:
            self.span_linear = torch.nn.Linear(hf_config.hidden_size * 2, config.linear_size)
            self.width_embeddings = None
        self.start_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.end_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.span_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))

        self.start_loss_weight = config.start_loss_weight
        self.end_loss_weight = config.end_loss_weight
        self.span_loss_weight = config.span_loss_weight
        self.threshold_loss_weight = config.threshold_loss_weight
        self.ner_loss_weight = config.ner_loss_weight
        self.similarity_loss = config.similarity_loss

        # Initialize weights and apply final processing
        self.post_init()

        self.text_encoder = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            config=hf_config,
        )
        self.type_encoder = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            config=hf_config,
        )

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def gradient_checkpointing_enable(self):
        self.text_encoder.gradient_checkpointing_enable()
        self.type_encoder.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        type_input_ids: torch.LongTensor = None,
        type_attention_mask: torch.Tensor = None,
        type_token_type_ids: torch.Tensor = None,
        ner: Optional[Dict] = None,
        return_dict: bool = None,
    ):
        return_dict = return_dict if return_dict is not None else self.hf_config.use_return_dict

        outputs = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        # batch_size x seq_length x hidden_size
        sequence_output = outputs[0]

        type_outputs = self.type_encoder(
            type_input_ids,
            attention_mask=type_attention_mask,
            token_type_ids=type_token_type_ids if type_token_type_ids is not None else None,
            return_dict=return_dict,
        )
        # num_types x hidden_size
        type_output = type_outputs[0][:, 0]

        batch_size, seq_length, _ = sequence_output.size()
        num_types, _ = type_output.size()

        # num_types x hidden_size
        type_start_output = F.normalize(self.dropout(self.type_start_linear(type_output)), dim=-1)
        type_end_output = F.normalize(self.dropout(self.type_end_linear(type_output)), dim=-1)
        # batch_size x seq_length x hidden_size
        sequence_start_output = F.normalize(self.dropout(self.start_linear(sequence_output)), dim=-1)
        sequence_end_output = F.normalize(self.dropout(self.end_linear(sequence_output)), dim=-1)

        # batch_size x num_types x seq_length
        start_scores = self.start_logit_scale.exp() * type_start_output.unsqueeze(0) @ sequence_start_output.transpose(1, 2)
        end_scores = self.end_logit_scale.exp() * type_end_output.unsqueeze(0) @ sequence_end_output.transpose(1, 2)

        # batch_size x seq_length x seq_length x hidden_size*2
        span_output = torch.cat(
            [
                sequence_output.unsqueeze(2).expand(-1, -1, seq_length, -1),
                sequence_output.unsqueeze(1).expand(-1, seq_length, -1, -1),
            ],
            dim=3
        )

        # span_width_embeddings
        if self.width_embeddings is not None:
            range_vector = torch.cuda.LongTensor(seq_length, device=sequence_output.device).fill_(1).cumsum(0) - 1
            span_width = range_vector.unsqueeze(0) - range_vector.unsqueeze(1) + 1
            # seq_length x seq_length x hidden_size
            span_width_embeddings = self.width_embeddings(span_width * (span_width > 0))
            span_output = torch.cat([
                span_output, span_width_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)], dim=3)

        # batch_size x seq_length x seq_length x hidden_size
        span_linear_output = F.normalize(
            self.dropout(self.span_linear(span_output)).view(batch_size, seq_length * seq_length, -1), dim=-1
        )
        # num_types x hidden_size
        type_linear_output = F.normalize(self.dropout(self.type_span_linear(type_output)), dim=-1)

        span_scores = self.span_logit_scale.exp() * type_linear_output.unsqueeze(0) @ span_linear_output.transpose(1, 2)
        span_scores = span_scores.view(batch_size, num_types, seq_length, seq_length)

        total_loss = None
        if ner is not None:
            flat_start_scores = start_scores.view(batch_size * num_types, seq_length)
            flat_end_scores = end_scores.view(batch_size * num_types, seq_length)
            flat_span_scores = span_scores.view(batch_size * num_types, seq_length, seq_length)
            start_negative_mask = ner["start_negative_mask"].view(batch_size * num_types, seq_length)
            end_negative_mask = ner["end_negative_mask"].view(batch_size * num_types, seq_length)
            span_negative_mask = ner["span_negative_mask"].view(batch_size * num_types, seq_length, seq_length)

            start_threshold_loss = contrastive_loss(flat_start_scores, 0, start_negative_mask)
            end_threshold_loss = contrastive_loss(flat_end_scores, 0, end_negative_mask)
            span_threshold_loss = contrastive_loss(flat_span_scores, (0, 0), span_negative_mask)

            threshold_loss = (
                self.start_loss_weight * start_threshold_loss +
                self.end_loss_weight * end_threshold_loss +
                self.span_loss_weight * span_threshold_loss
            )

            ner_indices = ner["example_indices"]
            ner_starts, ner_ends = ner["example_starts"], ner["example_ends"]
            ner_start_masks, ner_end_masks = ner["example_start_masks"], ner["example_end_masks"]
            ner_span_masks = ner["example_span_masks"]

            start_loss = contrastive_loss(start_scores[tuple(ner_indices)], ner_starts, ner_start_masks)
            end_loss = contrastive_loss(end_scores[tuple(ner_indices)], ner_ends, ner_end_masks)
            span_loss = contrastive_loss(span_scores[tuple(ner_indices)], (ner_starts, ner_ends), ner_span_masks)

            total_loss = (
                self.start_loss_weight * start_loss +
                self.end_loss_weight * end_loss +
                self.span_loss_weight * span_loss
            )

            total_loss = self.ner_loss_weight * total_loss + self.threshold_loss_weight * threshold_loss

        if not return_dict:
            output = (start_scores, end_scores, span_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BinderModelOutput(
            loss=total_loss,
            start_scores=start_scores,
            end_scores=end_scores,
            span_scores=span_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BinderDecoder(PreTrainedModel):
    config_class = BinderConfig

    def __init__(self, config):
        super().__init__(config)

        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            cache_dir=config.cache_dir,
            revision=config.revision,
            use_auth_token=config.use_auth_token,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )
        self.hf_config = hf_config
        self.config.pruned_heads = hf_config.pruned_heads
        self.dropout = torch.nn.Dropout(0.1)
        self.type_start_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        self.type_end_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        self.type_span_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        self.start_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        self.end_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        if config.use_span_width_embedding:
            self.span_linear = torch.nn.Linear(hf_config.hidden_size * 2 + config.linear_size, config.linear_size)
            self.width_embeddings = torch.nn.Embedding(config.max_span_width, config.linear_size, padding_idx=0)
        else:
            self.span_linear = torch.nn.Linear(hf_config.hidden_size * 2, config.linear_size)
            self.width_embeddings = None
        self.start_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.end_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.span_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))

        self.start_loss_weight = config.start_loss_weight
        self.end_loss_weight = config.end_loss_weight
        self.span_loss_weight = config.span_loss_weight
        self.threshold_loss_weight = config.threshold_loss_weight
        self.ner_loss_weight = config.ner_loss_weight

        # Initialize weights and apply final processing
        self.post_init()

        self.text_encoder = AutoModelForCausalLM.from_pretrained(
            config.pretrained_model_name_or_path,
            config=hf_config,
        ).model
        self.type_encoder = AutoModelForCausalLM.from_pretrained(
            config.pretrained_model_name_or_path,
            config=hf_config,
        ).model

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def gradient_checkpointing_enable(self):
        self.text_encoder.gradient_checkpointing_enable()
        self.type_encoder.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        type_input_ids: torch.LongTensor = None,
        type_attention_mask: torch.Tensor = None,
        type_token_type_ids: torch.Tensor = None,
        ner: Optional[Dict] = None,
        return_dict: bool = None,
    ):
        return_dict = return_dict if return_dict is not None else self.hf_config.use_return_dict

        outputs = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        # batch_size x seq_length x hidden_size
        sequence_output = outputs[0]

        type_outputs = self.type_encoder(
            type_input_ids,
            attention_mask=type_attention_mask,
            token_type_ids=type_token_type_ids if type_token_type_ids is not None else None,
            return_dict=return_dict,
        ).last_hidden_state

        mask = type_attention_mask
        mask = mask.unsqueeze(-1).to(type_outputs.dtype)  # (batch_size, seq_length, 1)
        masked_hidden = type_outputs * mask
        summed = masked_hidden.sum(dim=1)  # (batch_size, hidden_size)
        denom = mask.sum(dim=1).clamp(min=1e-6)  # (batch_size, 1)
        type_output = summed / denom  # (batch_size, hidden_size)

        batch_size, seq_length, _ = sequence_output.size()
        num_types, _ = type_output.size()

        # num_types x hidden_size
        type_start_output = F.normalize(self.dropout(self.type_start_linear(type_output)), dim=-1)
        type_end_output = F.normalize(self.dropout(self.type_end_linear(type_output)), dim=-1)
        # batch_size x seq_length x hidden_size
        sequence_start_output = F.normalize(self.dropout(self.start_linear(sequence_output)), dim=-1)
        sequence_end_output = F.normalize(self.dropout(self.end_linear(sequence_output)), dim=-1)

        # batch_size x num_types x seq_length
        start_scores = self.start_logit_scale.exp() * type_start_output.unsqueeze(0) @ sequence_start_output.transpose(1, 2)
        end_scores = self.end_logit_scale.exp() * type_end_output.unsqueeze(0) @ sequence_end_output.transpose(1, 2)

        # batch_size x seq_length x seq_length x hidden_size*2
        span_output = torch.cat(
            [
                sequence_output.unsqueeze(2).expand(-1, -1, seq_length, -1),
                sequence_output.unsqueeze(1).expand(-1, seq_length, -1, -1),
            ],
            dim=3
        )

        # span_width_embeddings
        if self.width_embeddings is not None:
            range_vector = torch.cuda.LongTensor(seq_length, device=sequence_output.device).fill_(1).cumsum(0) - 1
            span_width = range_vector.unsqueeze(0) - range_vector.unsqueeze(1) + 1
            # seq_length x seq_length x hidden_size
            span_width_embeddings = self.width_embeddings(span_width * (span_width > 0))
            span_output = torch.cat([
                span_output, span_width_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)], dim=3)

        # batch_size x seq_length x seq_length x hidden_size
        span_linear_output = F.normalize(
            self.dropout(self.span_linear(span_output)).view(batch_size, seq_length * seq_length, -1), dim=-1
        )
        # num_types x hidden_size
        type_linear_output = F.normalize(self.dropout(self.type_span_linear(type_output)), dim=-1)

        span_scores = self.span_logit_scale.exp() * type_linear_output.unsqueeze(0) @ span_linear_output.transpose(1, 2)
        span_scores = span_scores.view(batch_size, num_types, seq_length, seq_length)

        total_loss = None
        if ner is not None:
            flat_start_scores = start_scores.view(batch_size * num_types, seq_length)
            flat_end_scores = end_scores.view(batch_size * num_types, seq_length)
            flat_span_scores = span_scores.view(batch_size * num_types, seq_length, seq_length)
            start_negative_mask = ner["start_negative_mask"].view(batch_size * num_types, seq_length)
            end_negative_mask = ner["end_negative_mask"].view(batch_size * num_types, seq_length)
            span_negative_mask = ner["span_negative_mask"].view(batch_size * num_types, seq_length, seq_length)

            start_threshold_loss = contrastive_loss(flat_start_scores, 0, start_negative_mask)
            end_threshold_loss = contrastive_loss(flat_end_scores, 0, end_negative_mask)
            span_threshold_loss = contrastive_loss(flat_span_scores, (0, 0), span_negative_mask)

            threshold_loss = (
                self.start_loss_weight * start_threshold_loss +
                self.end_loss_weight * end_threshold_loss +
                self.span_loss_weight * span_threshold_loss
            )

            ner_indices = ner["example_indices"]
            ner_starts, ner_ends = ner["example_starts"], ner["example_ends"]
            ner_start_masks, ner_end_masks = ner["example_start_masks"], ner["example_end_masks"]
            ner_span_masks = ner["example_span_masks"]

            start_loss = contrastive_loss(start_scores[tuple(ner_indices)], ner_starts, ner_start_masks)
            end_loss = contrastive_loss(end_scores[tuple(ner_indices)], ner_ends, ner_end_masks)
            span_loss = contrastive_loss(span_scores[tuple(ner_indices)], (ner_starts, ner_ends), ner_span_masks)

            total_loss = (
                self.start_loss_weight * start_loss +
                self.end_loss_weight * end_loss +
                self.span_loss_weight * span_loss
            )

            total_loss = self.ner_loss_weight * total_loss + self.threshold_loss_weight * threshold_loss

        if not return_dict:
            output = (start_scores, end_scores, span_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BinderModelOutput(
            loss=total_loss,
            start_scores=start_scores,
            end_scores=end_scores,
            span_scores=span_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BinderFocalModel(PreTrainedModel):
    config_class = BinderFocalConfig

    def __init__(self, config):
        super().__init__(config)

        text_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config.text_encoder,
            hidden_dropout_prob=config.dropout,
        )
        type_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config.type_encoder,
            hidden_dropout_prob=config.dropout,
        )
        self.text_config = text_config
        self.type_config = type_config
        self.dropout = torch.nn.Dropout(config.dropout)
        # Two-layer feed-forward networks for each projection, with GELU activation and dropout in between
        self.type_start_ffn = torch.nn.Sequential(
            torch.nn.Linear(type_config.hidden_size, config.linear_size),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.linear_size, config.linear_size),
            torch.nn.Dropout(config.dropout),
        )
        self.type_end_ffn = torch.nn.Sequential(
            torch.nn.Linear(type_config.hidden_size, config.linear_size),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.linear_size, config.linear_size),
            torch.nn.Dropout(config.dropout),
        )
        self.type_span_ffn = torch.nn.Sequential(
            torch.nn.Linear(type_config.hidden_size, config.linear_size),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.linear_size, config.linear_size),
            torch.nn.Dropout(config.dropout),
        )
        self.text_start_ffn = torch.nn.Sequential(
            torch.nn.Linear(text_config.hidden_size, config.linear_size),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.linear_size, config.linear_size),
            torch.nn.Dropout(config.dropout),
        )
        self.text_end_ffn = torch.nn.Sequential(
            torch.nn.Linear(text_config.hidden_size, config.linear_size),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.linear_size, config.linear_size),
            torch.nn.Dropout(config.dropout),
        )
        if config.use_span_width_embedding:
            self.text_span_ffn = torch.nn.Sequential(
                torch.nn.Linear(text_config.hidden_size * 2 + config.linear_size, config.linear_size),
                torch.nn.GELU(),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(config.linear_size, config.linear_size),
                torch.nn.Dropout(config.dropout),
            )
            self.width_embeddings = torch.nn.Embedding(config.max_span_width, config.linear_size, padding_idx=0)
        else:
            self.text_span_ffn = torch.nn.Sequential(
                torch.nn.Linear(text_config.hidden_size * 2, config.linear_size),
                torch.nn.GELU(),
                torch.nn.Dropout(config.dropout),
                torch.nn.Linear(config.linear_size, config.linear_size),
                torch.nn.Dropout(config.dropout),
            )
            self.width_embeddings = None

        self.start_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.end_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.span_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))

        self.start_loss_weight = config.start_loss_weight
        self.end_loss_weight = config.end_loss_weight
        self.span_loss_weight = config.span_loss_weight

        # Initialize weights and apply final processing
        self.post_init()

        self.text_encoder = AutoModel.from_pretrained(
            config.text_encoder,
            config=text_config,
        )
        self.type_encoder = AutoModel.from_pretrained(
            config.type_encoder,
            config=type_config,
        )

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def gradient_checkpointing_enable(self):
        self.text_encoder.gradient_checkpointing_enable()
        self.type_encoder.gradient_checkpointing_enable()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def bce_with_mask(self, logits, targets, mask=None, pos_weight=None, focal_gamma=None, logit_adjust=None):
        """
        logits, targets, mask: same shape
        pos_weight: scalar tensor for BCE to upweight positives (handles imbalance)
        focal_gamma: if set (e.g., 1-2), applies focal term
        logit_adjust: scalar or tensor added to logits (e.g., logit adjustment for class prior)
        """
        if logit_adjust is not None:
            logits = logits + logit_adjust

        # BCE with logits (per-element)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=pos_weight
        )

        if focal_gamma is not None and focal_gamma > 0:
            # p = sigmoid(logit), pt = p if y=1 else (1-p)
            p = torch.sigmoid(logits)
            pt = torch.where(targets > 0, p, 1 - p)
            bce = ((1 - pt) ** focal_gamma) * bce

        if mask is not None:
            bce = bce * mask

        denom = (mask.sum() if mask is not None else torch.tensor(bce.numel(), device=bce.device, dtype=bce.dtype)).clamp_min(1)
        return bce.sum() / denom

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        type_input_ids: torch.LongTensor = None,
        type_attention_mask: torch.Tensor = None,
        type_token_type_ids: torch.Tensor = None,
        ner: Optional[Dict] = None,
        return_dict: bool = False,
    ):
        outputs = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        # batch_size x seq_length x hidden_size
        sequence_output = outputs[0]

        type_outputs = self.type_encoder(
            type_input_ids,
            attention_mask=type_attention_mask,
            token_type_ids=type_token_type_ids if type_token_type_ids is not None else None,
            return_dict=return_dict,
        )
        # num_types x hidden_size
        type_output = type_outputs[0][:, 0]
        # type_output = self.mean_pooling(type_outputs, type_attention_mask)

        batch_size, seq_length, _ = sequence_output.size()
        num_types, _ = type_output.size()

        # num_types x hidden_size
        type_start_output = F.normalize(self.dropout(self.type_start_ffn(type_output)), dim=-1)
        type_end_output = F.normalize(self.dropout(self.type_end_ffn(type_output)), dim=-1)
        # batch_size x seq_length x hidden_size
        sequence_start_output = F.normalize(self.dropout(self.text_start_ffn(sequence_output)), dim=-1)
        sequence_end_output = F.normalize(self.dropout(self.text_end_ffn(sequence_output)), dim=-1)

        # batch_size x num_types x seq_length
        start_scores = self.start_logit_scale.exp() * type_start_output.unsqueeze(0) @ sequence_start_output.transpose(1, 2)
        end_scores = self.end_logit_scale.exp() * type_end_output.unsqueeze(0) @ sequence_end_output.transpose(1, 2)

        # batch_size x seq_length x seq_length x hidden_size*2
        span_output = torch.cat(
            [
                sequence_output.unsqueeze(2).expand(-1, -1, seq_length, -1),
                sequence_output.unsqueeze(1).expand(-1, seq_length, -1, -1),
            ],
            dim=3
        )

        # span_width_embeddings
        if self.width_embeddings is not None:
            range_vector = torch.cuda.LongTensor(seq_length, device=sequence_output.device).fill_(1).cumsum(0) - 1
            span_width = range_vector.unsqueeze(0) - range_vector.unsqueeze(1) + 1
            # seq_length x seq_length x hidden_size
            span_width_embeddings = self.width_embeddings(span_width * (span_width > 0))
            span_output = torch.cat([
                span_output, span_width_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)], dim=3)

        # batch_size x seq_length x seq_length x hidden_size
        span_linear_output = F.normalize(
            self.dropout(self.text_span_ffn(span_output)).view(batch_size, seq_length * seq_length, -1), dim=-1
        )
        # num_types x hidden_size
        type_linear_output = F.normalize(self.dropout(self.type_span_ffn(type_output)), dim=-1)

        span_scores = self.span_logit_scale.exp() * type_linear_output.unsqueeze(0) @ span_linear_output.transpose(1, 2)
        span_scores = span_scores.view(batch_size, num_types, seq_length, seq_length)

        total_loss = None
        if ner is not None:
            start_targets = ner["start_targets"].to(dtype=start_scores.dtype, device=start_scores.device)
            end_targets = ner["end_targets"].to(dtype=end_scores.dtype, device=end_scores.device)
            span_targets = ner["span_targets"].to(dtype=span_scores.dtype, device=span_scores.device)

            start_mask = ner["start_valid_mask"].to(device=start_scores.device)
            end_mask = ner["end_valid_mask"].to(device=end_scores.device)
            span_mask = ner["span_valid_mask"].to(device=span_scores.device)

            # Optional: dynamic pos_weight per head (handles heavy imbalance)
            def _pos_weight(targets, mask):
                # (#neg / #pos) within valid region; clamp to avoid inf/0
                pos = (targets.bool() & mask).sum().clamp_min(1)
                neg = (mask.sum() - pos).clamp_min(1)
                return (neg.float() / pos.float()).detach()

            pos_weight_tok  = _pos_weight(start_targets, start_mask)  # same for end
            pos_weight_span = _pos_weight(span_targets, span_mask)

            focal_gamma = 1.0

            start_loss_bce = self.bce_with_mask(
                logits=start_scores, targets=start_targets, mask=start_mask,
                pos_weight=pos_weight_tok, focal_gamma=focal_gamma
            )
            end_loss_bce = self.bce_with_mask(
                logits=end_scores, targets=end_targets, mask=end_mask,
                pos_weight=pos_weight_tok, focal_gamma=focal_gamma
            )
            span_loss_bce = self.bce_with_mask(
                logits=span_scores, targets=span_targets, mask=span_mask,
                pos_weight=pos_weight_span, focal_gamma=focal_gamma
            )

            total_loss = (
                self.start_loss_weight * start_loss_bce +
                self.end_loss_weight   * end_loss_bce   +
                self.span_loss_weight  * span_loss_bce
            )

        if not return_dict:
            output = (start_scores, end_scores, span_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BinderModelOutput(
            loss=total_loss,
            start_scores=start_scores,
            end_scores=end_scores,
            span_scores=span_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )