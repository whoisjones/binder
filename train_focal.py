"""
Fine-tune Binder for named entity recognition.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.config import BinderFocalConfig
from src.model import BinderFocalModel
from src.trainer import BinderFocalDataCollator, BinderTrainer
from src import utils as postprocess_utils


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments for Binder.
    """
    model_class: str = field(
        metadata={"help": "Model class to use."}
    )
    text_encoder: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    type_encoder: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    text_tokenizer: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as text_encoder"}
    )
    type_tokenizer: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as type_encoder"}
    )
    text_config: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as text_encoder"}
    )
    type_config: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as type_encoder"}
    )
    dropout: float = field(
        default=0.1, metadata={"help": "Dropout rate for hidden states."}
    )
    use_span_width_embedding: bool = field(
        default=True, metadata={"help": "Use span width embeddings."}
    )
    linear_size: int = field(
        default=384, metadata={"help": "Size of the last linear layer."}
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

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
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
        default=16,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_span_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an entity span."
        },
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
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if sys.argv[-1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup env variables and logging
    os.environ["WANDB_PROJECT"] = "binder"
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
            model_name = model_args.text_encoder.split("/")[-1]
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

        train_samples = min(train_samples, len(raw_datasets["train"]))
        raw_datasets["train"] = raw_datasets["train"].select(range(train_samples))

    if training_args.do_eval and "validation" not in raw_datasets:
        if data_args.max_eval_samples is None:
            raise ValueError("max_eval_samples is required when validation file is not provided")

        train_valid_split = raw_datasets["train"].train_test_split(test_size=data_args.max_eval_samples)
        raw_datasets["train"] = train_valid_split["train"]
        raw_datasets["validation"] = train_valid_split["test"]

    id2label = {}
    if training_args.do_eval and "validation" in raw_datasets:
        dev_labels = set([label["type"] for labels in raw_datasets["validation"]['ner'] for label in labels])
        id2label['eval'] = {i: label for i, label in enumerate(dev_labels)}

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.text_tokenizer if model_args.text_tokenizer else model_args.text_encoder,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True,
    )
    
    config = BinderFocalConfig(
        text_encoder=model_args.config_name if model_args.config_name else model_args.text_encoder,
        type_encoder=model_args.type_encoder if model_args.type_encoder else model_args.type_encoder,
        dropout=model_args.dropout,
        max_span_width=data_args.max_seq_length + 1,
        use_span_width_embedding=model_args.use_span_width_embedding,
        linear_size=model_args.linear_size,
        start_loss_weight=model_args.start_loss_weight,
        end_loss_weight=model_args.end_loss_weight,
        span_loss_weight=model_args.span_loss_weight,
    )
    model = BinderFocalModel(config)

    def prepare_validation_features(examples, split: str = "eval"):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["text"],
            truncation=True,
            max_length=data_args.max_seq_length,
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
                remove_columns=raw_datasets["validation"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    # Data collator
    data_collator = BinderFocalDataCollator(tokenizer=tokenizer, id2label=id2label)

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