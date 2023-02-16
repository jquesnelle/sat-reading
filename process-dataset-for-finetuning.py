import argparse
import os
import sys
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

def main(args):
    save_dataset_path = f"{args.dataset}-processed"

    dataset = load_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=4096)

    dataset_columns = ["text", "answer", "requires", "id"]

    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["validation"]]).map(lambda x: tokenizer(
        x["text"], truncation=True), batched=True, remove_columns=dataset_columns)
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    tokenized_targets = concatenate_datasets([dataset["train"], dataset["validation"]]).map(lambda x: tokenizer(
        x["answer"], truncation=True), batched=True, remove_columns=dataset_columns)
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")

    def preprocess_function(sample,padding="max_length"):
        # add prefix to the input
        inputs = sample["text"]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample["answer"], max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset_columns)
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    tokenized_dataset["train"].save_to_disk(os.path.join(save_dataset_path, "train"))
    tokenized_dataset["validation"].save_to_disk(os.path.join(save_dataset_path, "eval"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model")
    parser.add_argument("dataset", type=str, help="Dataset")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    sys.exit(main(parse_args()))
