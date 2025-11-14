from datasets import Dataset

if __name__ == "__main__":
    dataset = Dataset.load_from_disk("sample/deu")
    for annotation in dataset[0]['spans_char']:
        print(f"{annotation['tag']}: {dataset[0]['text'][annotation['start']:annotation['end']]}")
    print(dataset[0])