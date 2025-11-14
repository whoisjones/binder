from datasets import load_dataset
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import DatasetDict

MAX_EVAL_SAMPLES_PER_DATASET = {
    "panx": 1000,
    "masakhaner": np.inf,
    "multinerd": np.inf,
    "multiconer_v1": 10000,
    "multiconer_v2": 10000,
    "dynamicner": np.inf,
    "uner": np.inf,
}

def process_single_split(dataset):
    column = "spans_char" if "spans_char" in dataset.column_names else "char_spans"
    df = dataset.to_pandas()
    avg_text_length = df['text'].apply(len).mean()
    avg_annotations_per_sentence = df[column].apply(len).mean()
    avg_distinct_annotations_per_sentence = df[column].apply(lambda x: len(set([span['tag'] for span in x]))).mean()
    distinct_entity_types = set(tag for tags in df[column].apply(lambda x: [span['tag'] for span in x]) for tag in tags)
    num_distinct_entity_types = len(distinct_entity_types)
    return {
        'num_sentences': len(df),
        'avg_text_length': avg_text_length,
        'avg_annotations_per_sentence': avg_annotations_per_sentence,
        'avg_distinct_annotations_per_sentence': avg_distinct_annotations_per_sentence,
        'distinct_entity_types': distinct_entity_types,
        'num_distinct_entity_types': num_distinct_entity_types,
    }

def compute_average(stats_per_language):
    distinct_entity_types = set().union(*stats_per_language['distinct_entity_types'])
    return {
        'num_sentences': stats_per_language['num_sentences'].sum(),
        'language': 'average',
        'avg_text_length': stats_per_language['avg_text_length'].mean(),
        'avg_annotations_per_sentence': stats_per_language['avg_annotations_per_sentence'].mean(),
        'avg_distinct_annotations_per_sentence': stats_per_language['avg_distinct_annotations_per_sentence'].mean(),
        'distinct_entity_types': distinct_entity_types,
        'num_distinct_entity_types': len(distinct_entity_types),
    }

def process_dataset_dict(dataset_dict):
    dataset_stats = []
    for language in dataset_dict:
        dataset_stats.append({"language": language, **process_single_split(dataset_dict[language])})
    dataset_stats = pd.DataFrame(dataset_stats)
    dataset_stats = pd.concat([dataset_stats, pd.DataFrame([compute_average(dataset_stats)])])
    return dataset_stats

import json
import glob

def compute_train():
    path = "/vol/tmp/goldejon/multilingual_ner/data/training_jsonl"
    path_tmp2 = "/vol/tmp2/goldejon/multilingual_ner/data/"
    dataset_stats = []
    for training_dataset in ["finerweb_4o_jsonl", "finerweb_gemma_jsonl", "finerweb_merged_jsonl"]:
        if "finerweb" in training_dataset:
            data_files = {
                d.split('.')[0]: 
                os.path.join(path_tmp2, training_dataset, d) for d in os.listdir(f"{path_tmp2}/{training_dataset}") if d.endswith(".jsonl")
            }
        else:
            data_files = {
                d.split('.')[0]: 
                os.path.join(path, training_dataset, d) for d in os.listdir(f"{path}/{training_dataset}") if d.endswith(".jsonl")
            }
        dataset = load_dataset("json", data_files=data_files)
        current_stats = process_dataset_dict(dataset)
        current_stats['dataset'] = training_dataset
        dataset_stats.append(current_stats)
    dataset_stats = pd.concat(dataset_stats)
    dataset_stats.to_csv("/vol/tmp/goldejon/multilingual_ner/analysis/train_dataset_stats.csv", index=False)
    dataset_stats = dataset_stats.drop(columns=['distinct_entity_types'])

    num_languages_per_dataset = (
        dataset_stats[dataset_stats['language'] != 'average']
        .groupby('dataset')['language']
        .nunique()
    )
    print("Number of languages per dataset (excluding 'average'):")
    print(num_languages_per_dataset)

    train_overview_table_for_paper = dataset_stats[dataset_stats['language'] == 'average']
    train_overview_table_for_paper = train_overview_table_for_paper.set_index('dataset').drop(columns=['language']).T

    train_overview_table_for_paper.loc['num_languages'] = num_languages_per_dataset[train_overview_table_for_paper.columns].values
    column_order = ['nuner', 'pilener', 'euro_glinerx', 'finerweb_4o_jsonl', "finerweb_gemma_jsonl", "finerweb_merged_jsonl"]
    train_overview_table_for_paper = train_overview_table_for_paper[column_order]

    column_paper_names = {
        'nuner': 'NuNER',
        'pilener': 'PileNER',
        'euro_glinerx': 'Euro-GLiNER-X',
        'finerweb_4o_jsonl': 'FiNERWeb (4O)',
        'finerweb_gemma_jsonl': 'FiNERWeb (Gemma)',
        'finerweb_merged_jsonl': 'FiNERWeb (Merged)',
    }

    row_paper_names = {
        'num_sentences': '\# Sentences',
        'avg_text_length': 'Avg. Text Length',
        'avg_annotations_per_sentence': 'Avg. Annotations/Sentence',
        'avg_distinct_annotations_per_sentence': 'Avg. Distinct Annotations/Sentence',
        'num_distinct_entity_types': 'Distinct Entity Types',
        'num_languages': '\# Languages',
    }

    train_overview_table_for_paper.rename(columns=column_paper_names, inplace=True)
    train_overview_table_for_paper.rename(index=row_paper_names, inplace=True)

    # Max 2 decimals
    train_overview_table_for_paper = train_overview_table_for_paper.round(2)
    # print out latex with 2 decimals
    print(train_overview_table_for_paper.to_latex(float_format="%.2f"))

def compute_eval():
    path = "/vol/tmp/goldejon/multilingual_ner/data/evaluation"
    dataset_stats = []
    for dataset_name in tqdm(os.listdir(path)):
        for language_dir in tqdm(os.listdir(path + f"/{dataset_name}")):
            language_code = language_dir.split("/")[-1].split(".")[0]

            dataset = DatasetDict.load_from_disk(f"{path}/{dataset_name}/{language_dir}")
            eval_split = "test" if "test" in dataset else "dev"
            test_split = dataset[eval_split]

            max_samples = MAX_EVAL_SAMPLES_PER_DATASET[dataset_name]
            if np.isinf(max_samples) or max_samples > len(test_split):
                test_split = test_split.shuffle(seed=42)
            else:
                test_split = test_split.shuffle(seed=42).select(range(int(max_samples)))

            current_stats = process_single_split(test_split)
            current_stats['dataset'] = dataset_name
            current_stats['language'] = language_code
            dataset_stats.append(current_stats)
    dataset_stats = pd.DataFrame(dataset_stats)
    dataset_stats.to_csv("/vol/tmp/goldejon/multilingual_ner/analysis/eval_dataset_stats.csv", index=False)

    agg_dataset_stats = dataset_stats.groupby(['dataset']).agg({
        'num_sentences': 'sum',
        'avg_text_length': 'mean',
        'avg_annotations_per_sentence': 'mean',
        'avg_distinct_annotations_per_sentence': 'mean',
        'distinct_entity_types': lambda x: list(set().union(*x)),
        'language': lambda x: len(x),
    })
    agg_dataset_stats['num_distinct_entity_types'] = agg_dataset_stats['distinct_entity_types'].apply(len)
    agg_dataset_stats = agg_dataset_stats.drop(columns=['distinct_entity_types']).rename({'language': 'num_languages'}).T

    column_paper_names = {
        'dynamicner': 'DynamicNER',
        'panx': 'PAN-X',
        'masakhaner': 'MasakhaNER',
        'multinerd': 'MultiNERD',
        'multiconer_v1': 'MultiCoNER v1',
        'multiconer_v2': 'MultiCoNER v2',
        'uner': 'U-NER',
    }

    row_paper_names = {
        'num_sentences': '\# Sentences',
        'avg_text_length': 'Avg. Text Length',
        'avg_annotations_per_sentence': 'Avg. Annotations/Sentence',
        'avg_distinct_annotations_per_sentence': 'Avg. Distinct Annotations/Sentence',
        'num_distinct_entity_types': 'Distinct Entity Types',
        'num_languages': '\# Languages',
    }

    agg_dataset_stats.rename(columns=column_paper_names, inplace=True)
    agg_dataset_stats.rename(index=row_paper_names, inplace=True)

    # Max 2 decimals
    agg_dataset_stats = agg_dataset_stats.round(2)
    # print out latex with 2 decimals
    print(agg_dataset_stats.to_latex(float_format="%.2f"))


if __name__ == "__main__":
    compute_train()
    # compute_eval()
