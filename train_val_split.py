import argparse

from datasets import load_dataset, DatasetDict

parser = argparse.ArgumentParser()
parser.add_argument("dataset_file")
parser.add_argument("val_set_size", type=int)
parser.add_argument("train_val_dataset_file")
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

dataset = load_dataset("json", data_files=args.dataset_file)
train_val = dataset["train"].train_test_split(test_size=args.val_set_size, shuffle=True, seed=args.seed)
split_dataset = DatasetDict({"train": train_val["train"], "validation": train_val["test"]})
split_dataset.save_to_disk(args.train_val_dataset_file)
