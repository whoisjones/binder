"""
Fine-tune Binder for named entity recognition.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.config import BinderConfig
from src.model import BinderDecoder
from src.collator import BinderDataCollator
from src.trainer import BinderDataCollator, BinderTrainer
from src import utils as postprocess_utils


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments for Binder.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    hidden_dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout rate for hidden states."}
    )
    use_span_width_embedding: bool = field(
        default=False, metadata={"help": "Use span width embeddings."}
    )
    linear_size: int = field(
        default=128, metadata={"help": "Size of the last linear layer."}
    )
    init_temperature: float = field(
        default=0.07, metadata={"help": "Init value of temperature used in contrastive loss."}
    )
    start_loss_weight: float = field(
        default=0.2, metadata={"help": "NER span start loss weight."}
    )
    end_loss_weight: float = field(
        default=0.2, metadata={"help": "NER span end loss weight."}
    )
    span_loss_weight: float = field(
        default=0.6, metadata={"help": "NER span loss weight."}
    )
    threshold_loss_weight: float = field(
        default=0.5, metadata={"help": "NER threshold loss weight."}
    )
    ner_loss_weight: float = field(
        default=0.5, metadata={"help": "NER loss weight."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use, from which it will decide entity types to use."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_span_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an entity span."
        },
    )
    entity_type_file: str = field(
        default=None,
        metadata={"help": "The entity type file contains all entity type names, descriptions, etc."},
    )
    dataset_entity_types: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "The entity types of this dataset, which are only a part of types in the entity type file."},
    )
    entity_type_key_field: Optional[str] = field(
        default="name",
        metadata={"help": "The field in the entity type file that will be used as key to sort entity types."},
    )
    entity_type_desc_field: Optional[str] = field(
        default="description",
        metadata={"help": "The field in the entity type file that corresponds to entity descriptions."},
    )
    prediction_postprocess_func: Optional[str] = field(
        default="postprocess_nested_predictions",
        metadata={"help": "The name of prediction postprocess function."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "The name of WANDB project."},
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if sys.argv[-1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup env variables and logging
    os.environ["WANDB_PROJECT"] = data_args.wandb_project or data_args.dataset_name
    os.environ["WANDB_DIR"] = training_args.output_dir
    os.makedirs(training_args.output_dir, exist_ok=True)
    log_file_handler = logging.FileHandler(os.path.join(training_args.output_dir, "run.log"), "a")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), log_file_handler],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    transformers.utils.logging.add_handler(log_file_handler)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    if isinstance(data_args.train_file, str) and data_args.train_file.endswith(".json"):
        data_files = {}
        if data_args.train_file is not None:
            extension = data_args.train_file.split(".")[-1]
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        for dataset_name in data_args.train_file:
            model_name = model_args.model_name_or_path.split("/")[-1]
            data_files[dataset_name] = Dataset.load_from_disk(f"/vol/tmp/goldejon/multilingual_ner/binder/data/training/tokenized/{model_name}/{dataset_name}")
        train_dataset = concatenate_datasets(data_files.values())
        raw_datasets = DatasetDict({"train": train_dataset})

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is None and 'train' not in raw_datasets:
            raise ValueError("max_train_samples is required when train file is not provided")
        train_samples = len(raw_datasets["train"]) if data_args.max_train_samples is None else data_args.max_train_samples
        if "validation" not in raw_datasets and training_args.do_eval and data_args.max_eval_samples is not None:
            train_samples += data_args.max_eval_samples
        if "test" not in raw_datasets and training_args.do_predict and data_args.max_predict_samples is not None:
            train_samples += data_args.max_predict_samples
        raw_datasets["train"] = raw_datasets["train"].select(range(train_samples))
    if training_args.do_eval and "validation" not in raw_datasets:
        if data_args.max_eval_samples is None:
            raise ValueError("max_eval_samples is required when validation file is not provided")
        train_valid_split = raw_datasets["train"].train_test_split(test_size=data_args.max_eval_samples)
        raw_datasets["train"] = train_valid_split["train"]
        raw_datasets["validation"] = train_valid_split["test"]

    if 'train' in raw_datasets and 'id' not in raw_datasets["train"].column_names:
        raw_datasets["train"] = raw_datasets["train"].add_column(
            name="id",
            column=[f"train_{i}" for i in range(len(raw_datasets["train"]))]
        )
    if 'validation' in raw_datasets and 'id' not in raw_datasets["validation"].column_names:
        raw_datasets["validation"] = raw_datasets["validation"].add_column(
            name="id",
            column=[f"eval_{i}" for i in range(len(raw_datasets["validation"]))]
        )

    id2label = {}
    if training_args.do_eval and "validation" in raw_datasets:
        dev_labels = set([label for labels in raw_datasets["validation"]['entity_types'] for label in labels])
        id2label['eval'] = {i: label for i, label in enumerate(dev_labels)}

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True,
    )
    
    config = BinderConfig(
        pretrained_model_name_or_path=model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
        max_span_width=data_args.max_seq_length + 1,
        use_span_width_embedding=model_args.use_span_width_embedding,
        linear_size=model_args.linear_size,
        init_temperature=model_args.init_temperature,
        start_loss_weight=model_args.start_loss_weight,
        end_loss_weight=model_args.end_loss_weight,
        span_loss_weight=model_args.span_loss_weight,
        threshold_loss_weight=model_args.threshold_loss_weight,
        ner_loss_weight=model_args.ner_loss_weight,
    )
    model = BinderDecoder(config)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Preprocessing the datasets.
    # Preprocessing is slightly different for training and evaluation.
    if training_args.do_train and "train" in raw_datasets:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        raise ValueError("No dataset provided")

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        processed_examples = {
            "input_ids": [],
            "attention_mask": [],
            "token_start_mask": [],
            "token_end_mask": [],
            "default_span_mask": [],
            "split": [],
            "example_id": [],
            "ner": [],
        }
        # RoBERTa doesn't need token_type_ids.
        if "token_type_ids" in tokenized_examples:
            processed_examples["token_type_ids"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]

            # Grab the sequence corresponding to that example (to know what is the text and what are special tokens).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # Start token index of the current text.
            text_start_index = 0
            while sequence_ids[text_start_index] != 0:
                text_start_index += 1

            # End token index of the current text.
            text_end_index = len(input_ids) - 1
            while sequence_ids[text_end_index] != 0:
                text_end_index -= 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]

            # Create token_start_mask and token_end_mask where mask = 1 if the corresponding token is either a start
            # or an end of a word in the original dataset.
            token_start_mask, token_end_mask = [], []
            word_start_chars = examples["word_start_chars"][sample_index]
            word_end_chars = examples["word_end_chars"][sample_index]
            for index, (start_char, end_char) in enumerate(offsets):
                if sequence_ids[index] != 0:
                    token_start_mask.append(0)
                    token_end_mask.append(0)
                else:
                    token_start_mask.append(int(start_char in word_start_chars))
                    token_end_mask.append(int(end_char in word_end_chars))

            # Subword span mask: 1 if the span is in the text, 0 otherwise.
            default_span_mask = [
                [
                    (j - i >= 0) * s * e for j, e in enumerate(token_end_mask)
                ]
                for i, s in enumerate(token_start_mask)
            ]

            annotations = []
            entity_types = examples["entity_types"][sample_index]
            entity_start_chars = examples["entity_start_chars"][sample_index]
            entity_end_chars = examples["entity_end_chars"][sample_index]

            assert len(entity_types) == len(entity_start_chars) == len(entity_end_chars)
            for entity_type, start_char, end_char in zip(entity_types, entity_start_chars, entity_end_chars):
                # Detect if the span is in the text.
                if offsets[text_start_index][0] <= start_char and offsets[text_end_index][1] >= end_char:
                    start_token_index, end_token_index = text_start_index, text_end_index
                    # Move the start_token_index and end_token_index to the two ends of the span.
                    # Note: we could go after the last offset if the span is the last word (edge case).
                    while start_token_index <= text_end_index and offsets[start_token_index][0] <= start_char:
                        start_token_index += 1
                    start_token_index -= 1

                    while offsets[end_token_index][1] >= end_char:
                        end_token_index -= 1
                    end_token_index += 1

                    # Inclusive start and end.
                    annotations.append({
                        "type": entity_type,
                        "start": start_token_index,
                        "end": end_token_index,
                    })

            if len(annotations) == 0:
                continue

            processed_examples["input_ids"].append(input_ids)
            if "token_type_ids" in tokenized_examples:
                processed_examples["token_type_ids"].append(tokenized_examples["token_type_ids"][i])
            processed_examples["attention_mask"].append(tokenized_examples["attention_mask"][i])
            processed_examples["token_start_mask"].append(token_start_mask)
            processed_examples["token_end_mask"].append(token_end_mask)
            processed_examples["default_span_mask"].append(default_span_mask)
            processed_examples['ner'].append(annotations)
            processed_examples["split"].append("train")
            processed_examples["example_id"].append(examples["id"][sample_index])

        return processed_examples

    if training_args.do_train and "input_ids" not in raw_datasets["train"].column_names:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    # Validation preprocessing
    def prepare_validation_features(examples, split: str = "eval"):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to spans of the text, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["split"] = []
        tokenized_examples["example_id"] = []
        tokenized_examples["token_start_mask"] = []
        tokenized_examples["token_end_mask"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            tokenized_examples["split"].append(split)

            # Grab the sequence corresponding to that example (to know what is the text and what are special tokens).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several texts, this is the index of the example containing this text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Create token_start_mask and token_end_mask where mask = 1 if the corresponding token is either a start
            # or an end of a word in the original dataset.
            token_start_mask, token_end_mask = [], []
            word_start_chars = examples["word_start_chars"][sample_index]
            word_end_chars = examples["word_end_chars"][sample_index]
            for index, (start_char, end_char) in enumerate(tokenized_examples["offset_mapping"][i]):
                if sequence_ids[index] != 0:
                    token_start_mask.append(0)
                    token_end_mask.append(0)
                else:
                    token_start_mask.append(int(start_char in word_start_chars))
                    token_end_mask.append(int(end_char in word_end_chars))

            tokenized_examples["token_start_mask"].append(token_start_mask)
            tokenized_examples["token_end_mask"].append(token_end_mask)

            # Set to None the offset_mapping that are not part of the text so it's easy to determine if a token
            # position is part of the text or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == 0 else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if training_args.do_eval and "input_ids" not in raw_datasets["validation"].column_names:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    # Data collator
    data_collator = BinderDataCollator(tokenizer=tokenizer, id2label=id2label)

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage=f"eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        metrics = getattr(postprocess_utils, data_args.prediction_postprocess_func)(
            examples=examples,
            features=features,
            predictions=predictions,
            id_to_type=id2label[stage],
            max_span_length=data_args.max_span_length,
            output_dir=training_args.output_dir if training_args.should_save else None,
            log_level=log_level,
            prefix=stage,
            tokenizer=tokenizer,
            train_file=data_args.train_file,
        )

        return metrics

    # Initialize our Trainer
    trainer = BinderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)] if training_args.do_eval else None,
        post_process_function=post_processing_function,
        compute_metrics=None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # with profiler_callback.profiler:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "model_name": "Binder"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()