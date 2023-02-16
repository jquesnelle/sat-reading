import os
import argparse
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
)
import sys
from datasets import load_from_disk
import torch

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("model", type=str,
                        help="Model id to use for training.")
    parser.add_argument("dataset", type=str,
                        help="Path to the already processed dataset.")
    parser.add_argument(
        "--repository_id", type=str, default=None, help="Hugging Face Repository id for uploading models"
    )
    parser.add_argument("--output_dir", type=str,
                        default="outputs", help="Directory to output to")
    # add training hyperparameters for epochs, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed to use for training.")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to deepspeed config file.")
    parser.add_argument("--gradient_checkpointing", type=bool,
                        default=True, help="Path to deepspeed config file.")
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=HfFolder.get_token(),
        help="Token to use for uploading models to Hugging Face Hub.",
    )
    args = parser.parse_args()
    return args


def main(args):
    # set seed
    set_seed(args.seed)

    # load dataset from disk and tokenizer
    train_dataset = load_from_disk(os.path.join(args.dataset, "train"))
    eval_dataset = load_from_disk(os.path.join(args.dataset, "eval"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        # this is needed for gradient checkpointing
        use_cache=False if args.gradient_checkpointing else True,
    )

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
    )

    # Define training args
    output_dir = os.path.join(args.output_dir, f'{args.model.split("/")[-1]}-{args.dataset.replace("/", "-")}')
    training_args = Seq2SeqTrainingArguments(
        overwrite_output_dir=True,
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        fp16=False,  # T5 overflows with fp16
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # push to hub parameters
        report_to="tensorboard",
        push_to_hub=True if args.repository_id else False,
        hub_strategy="every_save",
        hub_model_id=args.repository_id if args.repository_id else None,
        hub_token=args.hf_token,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

    # Save our tokenizer and create model card
    tokenizer.save_pretrained(output_dir)
    trainer.create_model_card()
    trainer.save_model(output_dir)
    # Push the results to the hub
    if args.repository_id:
        trainer.push_to_hub()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
