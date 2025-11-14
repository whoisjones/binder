from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import os
import numpy as np
from tqdm import tqdm
import re


MAX_EVAL_SAMPLES_PER_DATASET = {
    "panx": 1000,
    "masakhaner": np.inf,
    "multinerd": np.inf,
    "multiconer_v1": 10000,
    "multiconer_v2": 10000,
    "dynamicner": np.inf,
    "uner": np.inf,
}

WORD_PATTERN = re.compile(r'\S+')

# Use batched map for faster column creation
def process_batch(batch, batch_idx):
    batch_size = len(batch["text"])
    # batch_idx is a list of indices when using with_indices=True and batched=True
    ids = [f"train_{idx}" for idx in batch_idx]
    entity_start_chars = []
    entity_end_chars = []
    entity_types = []
    word_start_chars = []
    word_end_chars = []

    for i in range(batch_size):
        spans = batch["spans_char"][i]
        entity_start_chars.append([span['start'] for span in spans])
        entity_end_chars.append([span['end'] for span in spans])
        entity_types.append([span['tag'] for span in spans])

        text = batch["text"][i]
        starts = []
        ends = []
        for m in WORD_PATTERN.finditer(text):
            starts.append(m.start())
            ends.append(m.end())
        word_start_chars.append(starts)
        word_end_chars.append(ends)

    return {
        "id": ids,
        "entity_start_chars": entity_start_chars,
        "entity_end_chars": entity_end_chars,
        "entity_types": entity_types,
        "word_start_chars": word_start_chars,
        "word_end_chars": word_end_chars,
    }

def evaluation_datasets():
    for dataset_name in MAX_EVAL_SAMPLES_PER_DATASET.keys():
        for language in tqdm(os.listdir(f"/vol/tmp/goldejon/multilingual_ner/data/evaluation_translated/{dataset_name}")):
            if os.path.exists(f"/vol/tmp/goldejon/multilingual_ner/binder/data/evaluation_translated/{dataset_name}/{language}.json"):
                continue
            dataset = DatasetDict.load_from_disk(f"/vol/tmp/goldejon/multilingual_ner/data/evaluation_translated/{dataset_name}/{language}")
            eval_split = "test" if "test" in dataset else "dev"
            test_split = dataset[eval_split]

            assert 'tokens' in test_split.column_names, f"Tokens column not found in {language}"
            assert 'spans_tokens' in test_split.column_names, f"Spans tokens column not found in {language}"
            max_samples = MAX_EVAL_SAMPLES_PER_DATASET[dataset_name]
            if np.isinf(max_samples) or max_samples > len(test_split):
                test_split = test_split.shuffle(seed=42)
            else:
                test_split = test_split.shuffle(seed=42).select(range(int(max_samples)))

            test_split = test_split.map(
                process_batch,
                with_indices=True,
                batched=True,
                batch_size=1000,
                num_proc=1,
            )

            test_split = test_split.remove_columns(["tokens", "spans_tokens", "spans_char"])

            os.makedirs(f"/vol/tmp/goldejon/multilingual_ner/binder/data/evaluation_translated/{dataset_name}", exist_ok=True)
            test_split.to_json(f'/vol/tmp/goldejon/multilingual_ner/binder/data/evaluation_translated/{dataset_name}/{language}.json', orient="records", lines=True)

def train_datasets():
    # dataset_name = "euro_glinerx"
    # dataset = Dataset.load_from_disk(f"/vol/tmp/goldejon/multilingual_ner/data/training_hf/{dataset_name}")
    path = "/vol/tmp/goldejon/multilingual_ner/data/finerweb/finerweb_final/gemma"
    data_files = {
        d.split('.')[0]: 
        os.path.join(path, d) for d in os.listdir(path) if d.endswith(".jsonl")
    }
    dataset = load_dataset("json", data_files=data_files)
    dataset = concatenate_datasets(dataset.values())

    # Map in batches for speeds
    dataset = dataset.map(
        process_batch,
        with_indices=True,
        batched=True,
        batch_size=1000,  # adjust as needed for memory
        num_proc=8,
    )

    dataset = dataset.remove_columns(["tokens", "spans_tokens", "spans_char"])
    os.makedirs(f"/vol/tmp/goldejon/multilingual_ner/binder/data/training", exist_ok=True)
    dataset.to_json(f'/vol/tmp/goldejon/multilingual_ner/binder/data/training/gemma.json', orient="records", lines=True)

if __name__ == "__main__":
    train_datasets()
    # evaluation_datasets()