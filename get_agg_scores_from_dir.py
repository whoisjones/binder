import os
import json

def main():
    path = "/vol/tmp/goldejon/multilingual_ner/binder/evaluations/baseline/xlm-roberta-base/dynamicner"
    all_f1 = []
    all_precision = []
    all_recall = []
    for file in os.listdir(path):
        if file.endswith("results.json") and file != "all_results.json":
            with open(os.path.join(path, file), "r") as f:
                metrics = json.load(f)
            all_f1.append(metrics["test_f1"])
            all_precision.append(metrics["test_precision"])
            all_recall.append(metrics["test_recall"])
    print(f"F1: {sum(all_f1) / len(all_f1)}")
    print(f"Precision: {sum(all_precision) / len(all_precision)}")
    print(f"Recall: {sum(all_recall) / len(all_recall)}")

if __name__ == "__main__":
    main()